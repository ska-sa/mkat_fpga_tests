/* Copyright 2016 SKA South Africa
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV

#include <spead2/common_raw_packet.h>
#include <spead2/send_udp_ibv.h>

namespace spead2
{
namespace send
{

constexpr std::size_t udp_ibv_stream::default_buffer_size;
constexpr int udp_ibv_stream::default_max_poll;
static constexpr int header_length =
    ethernet_frame::min_size + ipv4_packet::min_size + udp_packet::min_size;

ibv_qp_t udp_ibv_stream::create_qp(
    const ibv_pd_t &pd, const ibv_cq_t &send_cq, const ibv_cq_t &recv_cq, std::size_t n_slots)
{
    ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = send_cq.get();
    attr.recv_cq = recv_cq.get();
    attr.qp_type = IBV_QPT_RAW_PACKET;
    attr.cap.max_send_wr = n_slots;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.sq_sig_all = true;
    return ibv_qp_t(pd, &attr);
}

void udp_ibv_stream::reap()
{
    ibv_wc wc;
    int done;
    while ((done = send_cq.poll(1, &wc)) > 0)
    {
        if (wc.status != IBV_WC_SUCCESS)
        {
            log_warning("Work Request failed with code %1%", wc.status);
        }
        slot *s = &slots[wc.wr_id];
        available.push_back(s);
    }
}

udp_ibv_stream::rerun_async_send_packet::rerun_async_send_packet(
    udp_ibv_stream *self,
    const packet &pkt,
    udp_ibv_stream::completion_handler &&handler)
    : self(self), pkt(&pkt), handler(std::move(handler))
{
}

void udp_ibv_stream::rerun_async_send_packet::operator()(
    boost::system::error_code ec, std::size_t bytes_transferred)
{
    (void) bytes_transferred;
    if (ec)
    {
        handler(ec, 0);
    }
    else
    {
        ibv_cq *event_cq;
        void *event_cq_context;
        // This should be non-blocking, since we were woken up
        self->comp_channel.get_event(&event_cq, &event_cq_context);
        self->send_cq.ack_events(1);
        self->async_send_packet(*pkt, std::move(handler));
    }
}

udp_ibv_stream::invoke_handler::invoke_handler(
    udp_ibv_stream::completion_handler &&handler,
    boost::system::error_code ec,
    std::size_t bytes_transferred)
    : handler(std::move(handler)), ec(ec), bytes_transferred(bytes_transferred)
{
}

void udp_ibv_stream::invoke_handler::operator()()
{
    handler(ec, bytes_transferred);
}

void udp_ibv_stream::async_send_packet(const packet &pkt, completion_handler &&handler)
{
    try
    {
        reap();
        if (available.empty())
        {
            if (comp_channel)
            {
                send_cq.req_notify(false);
                // Need to check again, in case of a race
                reap();
                comp_channel_wrapper.async_read_some(
                    boost::asio::null_buffers(),
                    rerun_async_send_packet(this, pkt, std::move(handler)));
                return;
            }
            else
            {
                // Poll mode - keep trying until we have space
                while (available.empty())
                    reap();
            }
        }
        slot *s = available.back();
        available.pop_back();

        std::size_t payload_size = boost::asio::buffer_size(pkt.buffers);
        ipv4_packet ipv4 = s->frame.payload_ipv4();
        ipv4.total_length(payload_size + udp_packet::min_size + ipv4.header_length());
        ipv4.update_checksum();
        udp_packet udp = ipv4.payload_udp();
        udp.length(payload_size + udp_packet::min_size);
        packet_buffer payload = udp.payload();
        boost::asio::buffer_copy(boost::asio::mutable_buffer(payload), pkt.buffers);
        s->sge.length = payload_size + (payload.data() - s->frame.data());
        qp.post_send(&s->wr);
        get_io_service().post(invoke_handler(std::move(handler), boost::system::error_code(),
                                             payload_size));
    }
    catch (std::system_error &e)
    {
        get_io_service().post(invoke_handler(std::move(handler),
            boost::system::error_code(e.code().value(), boost::system::system_category()), 0));
    }
}

udp_ibv_stream::udp_ibv_stream(
    io_service_ref io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    const stream_config &config,
    const boost::asio::ip::address &interface_address,
    std::size_t buffer_size,
    int ttl,
    int comp_vector,
    int max_poll)
    : stream_impl<udp_ibv_stream>(std::move(io_service), config),
    n_slots(std::max(std::size_t(1), buffer_size / (config.get_max_packet_size() + header_length))),
    max_poll(max_poll),
    socket(get_io_service(), endpoint.protocol()),
    cm_id(event_channel, nullptr, RDMA_PS_UDP),
    comp_channel_wrapper(get_io_service())
{
    if (!endpoint.address().is_v4() || !endpoint.address().is_multicast())
        throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    if (max_poll <= 0)
        throw std::invalid_argument("max_poll must be positive");
    socket.bind(boost::asio::ip::udp::endpoint(interface_address, 0));
    // Re-compute buffer_size as a whole number of slots
    const std::size_t max_raw_size = config.get_max_packet_size() + header_length;
    buffer_size = n_slots * max_raw_size;

    cm_id.bind_addr(interface_address);
    pd = ibv_pd_t(cm_id);
    if (comp_vector >= 0)
    {
        comp_channel = ibv_comp_channel_t(cm_id);
        comp_channel_wrapper = comp_channel.wrap(get_io_service());
        send_cq = ibv_cq_t(cm_id, n_slots, nullptr,
                           comp_channel, comp_vector % cm_id->verbs->num_comp_vectors);
    }
    else
        send_cq = ibv_cq_t(cm_id, n_slots, nullptr);
    recv_cq = ibv_cq_t(cm_id, 1, nullptr);
    qp = create_qp(pd, send_cq, recv_cq, n_slots);
    qp.modify(IBV_QPS_INIT, cm_id->port_num);
    qp.modify(IBV_QPS_RTR);
    qp.modify(IBV_QPS_RTS);

    std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
    buffer = allocator->allocate(max_raw_size * n_slots, nullptr);
    mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
    slots.reset(new slot[n_slots]);
    mac_address destination_mac = multicast_mac(endpoint.address());
    mac_address source_mac = interface_mac(interface_address);
    for (std::size_t i = 0; i < n_slots; i++)
    {
        slots[i].frame = ethernet_frame(buffer.get() + i * max_raw_size, max_raw_size);
        slots[i].sge.addr = (uintptr_t) slots[i].frame.data();
        slots[i].sge.lkey = mr->lkey;
        slots[i].wr.sg_list = &slots[i].sge;
        slots[i].wr.num_sge = 1;
        slots[i].wr.opcode = IBV_WR_SEND;
        slots[i].wr.wr_id = i;
        slots[i].frame.destination_mac(destination_mac);
        slots[i].frame.source_mac(source_mac);
        slots[i].frame.ethertype(ipv4_packet::ethertype);
        ipv4_packet ipv4 = slots[i].frame.payload_ipv4();
        ipv4.version_ihl(0x45);  // IPv4, 20 byte header
        // total_length will change later to the actual packet size
        ipv4.total_length(config.get_max_packet_size() + ipv4_packet::min_size + udp_packet::min_size);
        ipv4.flags_frag_off(ipv4_packet::flag_do_not_fragment);
        ipv4.ttl(ttl);
        ipv4.protocol(udp_packet::protocol);
        ipv4.source_address(interface_address.to_v4());
        ipv4.destination_address(endpoint.address().to_v4());
        udp_packet udp = ipv4.payload_udp();
        udp.source_port(socket.local_endpoint().port());
        udp.destination_port(endpoint.port());
        udp.length(config.get_max_packet_size() + udp_packet::min_size);
        udp.checksum(0);
        available.push_back(&slots[i]);
    }
}

udp_ibv_stream::~udp_ibv_stream()
{
    /* Wait until we have confirmation that all the packets
     * have been put on the wire before tearing down the data
     * structures.
     */
    flush();
    while (available.size() < n_slots)
        reap();
}

} // namespace send
} // namespace spead2

#endif // SPEAD2_USE_IBV
