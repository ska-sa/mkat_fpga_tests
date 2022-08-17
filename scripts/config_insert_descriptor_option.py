#!/usr/bin/env python

import argcomplete
import argparse
import fileinput
import re
import sys
import glob

def main():
    parser = argparse.ArgumentParser(description='Insert send_descriptor option in instrument config files '
                                                 'for each data product found.')
    parser.add_argument('-d', '--config_dir', help='Specify config file directory',
                        required=True)
    argcomplete.autocomplete(parser)
    args = vars(parser.parse_args())

    conf_dict = {
        'antenna-channelised-voltage':      'false',
        'baseline-correlation-products':    'true',
        'tied-array-channelised-voltage.0x':'false',
        'tied-array-channelised-voltage.0y':'false',
        'tied-array-channelised-voltage.1x':'false',
        'tied-array-channelised-voltage.1y':'false',
        'tied-array-channelised-voltage.2x':'false',
        'tied-array-channelised-voltage.2y':'false',
        'tied-array-channelised-voltage.3x':'false',
        'tied-array-channelised-voltage.3y':'false'
    }
    try:
        config_dir = args.get("config_dir", False)
        assert config_dir
        list_of_files = glob.glob('{}/*'.format(config_dir))
        config_files = filter(None, map(lambda f: f if (re.search('bc(.*)n', f)) else None, list_of_files))
    except Exception as e:
        sys.exit(e.message)
    else:
        for config_file in config_files:
            with open(config_file) as f:
                contents = f.readlines()
                contents = [''.join(x.split(' ')) for x in contents]
                for product in conf_dict.keys():
                    prod_idx_raw = [i for i,s in enumerate(contents) if product in s]
                    prod_idx = []
                    for idx in prod_idx_raw:
                        if '#' == contents[idx].replace(' ','')[0]:
                            # Commented line, ignore
                            pass
                        else:
                            prod_idx.append(idx)
                    if len(prod_idx) == 1:
                        prod_idx = prod_idx[0]
                        section_end_idx = [i+prod_idx for i,s in enumerate(contents[prod_idx:]) if '[' in s]
                        if len(section_end_idx) == 0:
                            section_end_idx = len(contents)
                        else:
                            section_end_idx = section_end_idx[0]
                        dest_base_idx_raw = [i+prod_idx for i,s in enumerate(contents[prod_idx:section_end_idx])
                            if 'output_destinations_base' in s]
                        dest_base_idx = []
                        for i in dest_base_idx_raw:
                            if '#' in contents[i].split('=')[0]:
                                pass
                            else:
                                dest_base_idx.append(i)
                        if len(dest_base_idx) > 1:
                            print ('Error: more than one output_destinations_base found for a section in {}'
                                ''.format(config_file))
                            sys.exit()
                        elif len(dest_base_idx) == 0:
                            print ('Error: output_destinations_base not found for a section in {}'
                                ''.format(config_file))
                            sys.exit()
                        else:
                            dest_base_idx = dest_base_idx[0]

                        if contents[dest_base_idx+1].split('=')[0] != 'send_descriptors':
                            contents.insert(dest_base_idx+1, 'send_descriptors={}\n'.format(conf_dict[product]))
                        else:
                            contents[dest_base_idx+1] = 'send_descriptors={}\n'.format(conf_dict[product])
                    elif len(prod_idx) == 0:
                        pass
                    else:
                        print ('Error: more than one data_product descriptor found in a section: {}'
                               ''.format(config_file))
                        sys.exit()

                #sanity check
                check_prod_idx_raw = [i for i,s in enumerate(contents) if 'output_products' in s]
                check_prod_idx = []
                for i in check_prod_idx_raw:
                    if '#' == contents[i].replace(' ','')[0]:
                        # Commented line, ignore
                        pass
                    else:
                        check_prod_idx.append(i)
                send_dsc_idx = [i for i,s in enumerate(contents) if 'send_descriptors' in s]
                if len(send_dsc_idx) != len(check_prod_idx):
                    print 'Error: send descriptor feilds not equal to output products: {}'.format(config_file)
                    sys.exit()

                for idx, line in enumerate(contents):
                    i = line.find('=')
                    if i != -1:
                        contents[idx] = line[:i]+' '+line[i]+' '+line[i+1:]

            with open(config_file, 'w') as f:
                f.write(''.join(contents))

if __name__ == "__main__":
    main()
