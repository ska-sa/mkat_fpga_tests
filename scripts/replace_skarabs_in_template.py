#!/usr/bin/env python

# Brutal way of replacing fhosts,xhosts and dhost in template(s)
import argparse
import argcomplete
import sys
import fileinput

def get_skarabs_from_leaf(leafs):
    dnsmasq_leases = '/var/lib/misc/dnsmasq.leases'
    with open(dnsmasq_leases) as f:
        dnsmasq_leases_data = f.readlines()
    skarabs = set([host for leaf in leafs
                   for i in dnsmasq_leases_data if ('100.%s' % leaf) in i
                   for host in i.split() if host.startswith('skarab')])
    return list(skarabs) if skarabs is not None else False

def main():
    parser = argparse.ArgumentParser(description='Simplified way of replacing fhosts, xhosts and dhost '
                                                 'from config templates')
    parser.add_argument('-l','--leaf', help='Specify which LEAF contains the skarabs needed',
                        required=True, nargs='+')
    parser.add_argument('-f','--config', help='Specify which config file, look in /etc/corr/templates',
                        required=True)
    parser.add_argument('-i','--inputs', help='Specify the number of inputs', required=True,)
    argcomplete.autocomplete(parser)
    args = vars(parser.parse_args())


    try:
        which_leaf = args.get('leaf', False)
        assert which_leaf
        which_leaf = [int(i) for i in which_leaf]
    except Exception as e:
        sys.exit(e.message)

    try:
        n_inputs = args.get("inputs", False)
        assert n_inputs
        n_inputs = int(n_inputs)
    except Exception as e:
        sys.exit(e.message)

    try:
        skarab_list = get_skarabs_from_leaf(which_leaf)
        assert skarab_list
        skarab_list = sorted(skarab_list)
    except Exception as e:
        sys.exit(e.message)

    try:
        config = args.get("config", False)
        assert config
        assert len(skarab_list[:n_inputs]) == n_inputs
        fhosts = 'hosts = %s\n' % ','.join(skarab_list[:n_inputs])
        assert len(skarab_list[n_inputs:n_inputs*2]) == n_inputs
        xhosts = 'hosts = %s\n' % ','.join(skarab_list[n_inputs:n_inputs*2])
        dhost = 'host = %s\n' % skarab_list[-1]
        print 'fhost: ', fhosts
        print 'xhost: ', xhosts
        print 'dhost: ', dhost
    except Exception as e:
        sys.exit(e.message)
    else:
        # Brutal search and replace
        with open(config) as f:
           config_file = f.readlines()
        fengine_ind = config_file.index('[fengine]\n') + 1
        xengine_ind = config_file.index('[xengine]\n') + 1
        dengine_ind = config_file.index('[dsimengine]\n') + 1
        config_file[fengine_ind:fengine_ind+1] = [fhosts]
        config_file[xengine_ind:xengine_ind+1] = [xhosts]
        config_file[config_file.index(
            [i for i in config_file[dengine_ind:] if 'host = ' in i][0])] = dhost
        with open(config, 'w') as f:
            f.write(''.join(config_file))

        # for line in fileinput.input([config], inplace=False):
        #     if line.strip().startswith('hosts = '):
        #         line = fhosts
        #     if line.strip().startswith('hosts = '):
        #         line = xhosts
        #     if line.strip().startswith('host = '):
        #         line = dhost
        #     sys.stdout.write(line)

if __name__ == "__main__":
    main()