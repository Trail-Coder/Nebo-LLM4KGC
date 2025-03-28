input_file = ''
output_file = ''

def add_line_numbers(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for index, line in enumerate(lines, start=0):
            file.write(f"{index}\t{line}")

add_line_numbers(input_file, output_file)