import os
import xlsxwriter as excel


class LoggingSheet:
    def __init__(self, sheet_name, column_names=[]):
        self.name = sheet_name
        self.columns = {c_name: [] for c_name in column_names}


    # get the logging sheet's name
    def get_name(self):
        return self.name


    # get all the columns in the logging sheet as a dictionary
    def get_columns(self):
        return self.columns


    # add one new column in to the columns' dictionary
    def add_column(self, column_name):
        self.columns[column_name] = []


    # add multiple new column in to the columns' dictionary
    def add_columns(self, column_names):
        self.columns.update({c_name: [] for c_name in column_names})


    # append a value to a specific column
    def add_value(self, column_name, value):
        assert column_name in self.columns, 'Column name not exist!'
        self.columns[column_name].append(value)


class Logging:
    def __init__(self, sheet_names = []):
        self.sheets = {sheet_name: LoggingSheet(sheet_name) for sheet_name in sheet_names}


    def __getitem__(self, sheet_name):
        return self.sheets[sheet_name]


    # add a new LoggingSheet object in to the sheets dictionary
    def add_sheet(self, sheet_name, column_names = []):
        self.sheets[sheet_name] = LoggingSheet(sheet_name, column_names)


    # record logging data in an Excel file
    # parameter: log_dict = {worksheet_name: {field_name: list_of_data}}
    def save(self, log_dir, file_name ='log.xlsx', order = []):
        workbook = excel.Workbook(os.path.join(log_dir, file_name))
        for sheet in self.sheets.values():
            worksheet = workbook.add_worksheet(sheet.get_name())

            data_dictionary = sheet.get_columns()

            if len(order) == 0:
                columns_list = data_dictionary
            else:
                columns_list = order

            column = 0
            for column_name in columns_list:
                row = 0
                worksheet.write(row, column, column_name)
                for value in data_dictionary[column_name]:
                    row += 1
                    worksheet.write(row, column, value)
                column += 1
        workbook.close()


# check and create directory
def make_dir(dir_path, allow_repeat=False):
    if allow_repeat:
        index = 1
        while True:
            new_path = os.path.join(os.path.dirname(dir_path), '{}_{}'.format(os.path.basename(dir_path), index))
            if os.path.exists(new_path):
                index += 1
                continue
            else:
                os.mkdir(new_path)
                return new_path
    else:
        if not(os.path.exists(dir_path)):
            os.mkdir(dir_path)
        return dir_path
