input_filename = ''
output_filename = ''

with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
    for line in infile:
        elements = line.strip().split('\t')
        if elements[-1] == '1':
            elements = elements[:-1]
            outfile.write('\t'.join(elements) + '\n')