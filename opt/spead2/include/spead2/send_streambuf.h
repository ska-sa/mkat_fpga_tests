/* Copyright 2015 SKA South Africa
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

#ifndef SPEAD2_SEND_STREAMBUF_H
#define SPEAD2_SEND_STREAMBUF_H

#include <streambuf>
#include <functional>
#include <boost/asio.hpp>
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

/**
 * Puts packets into a streambuf (which could come from an @c ostream). This
 * should not be used for a blocking stream such as a wrapper around TCP,
 * because doing so will block the asio handler thread.
 */
class streambuf_stream : public stream_impl<streambuf_stream>
{
private:
    friend class stream_impl<streambuf_stream>;
    std::streambuf &streambuf;

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        std::size_t size = 0;
        for (const auto &buffer : pkt.buffers)
        {
            std::size_t buffer_size = boost::asio::buffer_size(buffer);
            // TODO: handle errors
            streambuf.sputn(boost::asio::buffer_cast<const char *>(buffer), buffer_size);
            size += buffer_size;
        }
        get_io_service().dispatch(std::bind(std::move(handler), boost::system::error_code(), size));
    }

public:
    /// Constructor
    streambuf_stream(
        io_service_ref io_service,
        std::streambuf &streambuf,
        const stream_config &config = stream_config());
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAMBUF_H
