import csv

input_file='winequality-white.csv'
output_file='winequality-white-normalized.csv'
min_max_values = dict()
csv_delimiter = ';'
discarded_column = 'quality'
discarded_column_index = 0

# First dataset iteration to find minimum and maximum values per column
with open(input_file, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=csv_delimiter)
    print('Getting the minimum and maximum values from "%s"... '% input_file, end='', flush=True)
    first_row_processed = False

    for row in csv_reader:
        # Initializes min and max values for each header column
        if not first_row_processed:
            index = 0
            for column in row:
                if (column == discarded_column):
                    discarded_column_index = index
                min_max_values[column] = [999999.0, -999999.0] # will store the min and the max values of the column
                index+=1
            first_row_processed = True

        # Updates min and max values for each column
        for column in row:
            value = float(row[column])
            if value < min_max_values[column][0]:
                min_max_values[column][0] = value # update the min value for this column
            elif value > min_max_values[column][1]:
                min_max_values[column][1] = value # update the max value for this column
    print('done', flush=True)

# Second dataset iteration to normalize and save values in the output file
with open(input_file, mode='r') as csv_read_file:
    csv_reader = csv.DictReader(csv_read_file, delimiter=csv_delimiter)
    print('Copying dataset with normalized values to "%s"... '% output_file, end='', flush=True)
    first_row_processed = False
    min_max_values_row_processed = False

    with open(output_file, 'w', newline='') as csv_write_file:
        csv_writer = csv.writer(csv_write_file, delimiter=csv_delimiter, quoting=csv.QUOTE_NONE)
        for read_row in csv_reader:
            # Write header row
            if not first_row_processed:
                csv_writer.writerow(list(read_row.keys())) # Component names
                first_row_processed = True

            if not min_max_values_row_processed:
                min_values_row = list()
                max_values_row = list()
                for index, column in enumerate(read_row):
                    min_values_row.append(min_max_values[column][0])
                    max_values_row.append(min_max_values[column][1])
                csv_writer.writerow(min_values_row) # Min values
                csv_writer.writerow(max_values_row) # Max values
                min_max_values_row_processed = True

            # Write rows with values
            normalized_values = list()
            for index, column in enumerate(read_row):
                value = float(read_row[column])
                if (index != discarded_column_index):
                    min_value = min_max_values[column][0]
                    max_value = min_max_values[column][1]
                    normalized_values.append((value - min_value) / (max_value - min_value)) # Normalize value between 0 and 1
                else:
                    normalized_values.append(value)
            csv_writer.writerow(normalized_values)
    print('done', flush=True)