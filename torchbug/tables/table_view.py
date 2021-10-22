from rich import print
from rich.table import Table
from rich.console import Console


class TableView(object):
    def __init__(self, data, name, last_columns=None):
        self.data = data
        self.name = name

        if last_columns is None:
            self.last_columns = ["n_params_grad", "count"]
        else:
            self.last_columns = last_columns

        self.table = self._create_table()
        self.console = Console()

    def _create_table(self):
        table = Table(show_header=True, header_style="bold magenta")
        columns = self._get_column_names(self.data)

        # Create columns
        for col in columns:
            if col == "type":
                table.add_column(col, justify="right", min_width=33)
            else:
                table.add_column(col, justify="right")

        # Add rows
        for item in self.data:
            row = ["--"] * len(table.columns)
            keys = item.keys()

            for idx in range(len(table.columns)):
                key = columns[idx]
                if key in keys:
                    row[idx] = str(item[key])

                if key == "type":
                    row[idx] = "[green]" + row[idx] + "[/green]"

            table.add_row(*row)

        return table

    def _get_column_names(self, data):
        columns = set()
        for row in data:
            for key in row.keys():
                if key not in ["type"] + self.last_columns:
                    columns.add(key)

        columns = sorted(list(columns))

        if columns:
            return ["type"] + columns + self.last_columns
        else:
            return columns

    def print(self):
        self.console.print(self.table)
