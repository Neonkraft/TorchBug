from collections import defaultdict
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from rich import print
from ..tables.table_view import TableView
from ..utils.utils import (
    find_mismatches,
    get_module_attrs,
    get_module_name,
    get_leaf_modules,
    show_comparison
)


class ModelStatistics(object):

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self._db = TinyDB(storage=MemoryStorage)

        self._init_db()

    def _init_db(self):
        leaf_modules = get_leaf_modules(self.model)

        for module in leaf_modules:
            self._add_module_to_db(module)

    def _create_row(self, module, n_params=True):
        module_name = get_module_name(module)
        module_attrs = get_module_attrs(module)

        row = {}
        row['type'] = module_name

        for attr in module_attrs:
            row[attr] = getattr(module, attr)

        if n_params:
            row['n_params_no_grad'] = sum([p.numel() for p in module.parameters()])
            row['n_params_grad'] = sum([p.numel() for p in module.parameters() if p.requires_grad])

        return row

    def _add_module_to_db(self, module):
        row = self._create_row(module)
        result = self._db.search(Query().fragment(row))

        assert len(result) == 0 or len(result) == 1, f"More than one results found for query: {row}"

        if result == []:
            row['count'] = 1
            self._db.insert(row)
        else:
            self._db.upsert({'count': result[0]['count'] + 1}, Query().fragment(row))

    def get_db(self):
        return self._db

    def query_for_module(self, module):
        row = self._create_row(module)
        return self._db.search(Query().fragment(row))

    def compare(self, other, as_table=True):
        other_db = other.get_db()

        db_rows = [row for row in self._db.all()]
        other_db_rows = [row for row in other_db.all()]

        mismatches_other = find_mismatches(db_rows, other_db_rows)
        mismatches_self = find_mismatches(other_db_rows, db_rows)

        missing_modules = dict()
        missing_modules[self.name] = mismatches_self
        missing_modules[other.name] = mismatches_other

        if mismatches_other or mismatches_self:
            print("\n[bold][red]Number of registered leaf modules do not match! See below:[/red][/bold]")
        else:
            print("\n[bold][green]Number of registered leaf modules match![/green][/bold]")

        show_comparison(missing_modules, as_table)
        return missing_modules

    def print(self, as_table=True, modules=None):
        print(f"\n[bold][magenta]Summary of {self.name}[/magenta][/bold]")

        all_rows = self._db.all()
        all_rows = sorted(all_rows, key=lambda x: str(x))

        row_types = defaultdict(lambda: [])

        for row in all_rows:
            if modules is None or (modules is not None and row["type"] in modules):
                row_types[row["type"]].append(row)

        for rows in row_types.values():
            table = TableView(rows, "")

            if as_table:
                table.print()
            else:
                print(table.data)

        n_params_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_params_no_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad == False)

        print(f"{len(all_rows)} different types of leaf modules")
        print(f"{n_params_grad} parameters (requires_grad = True)")
        print(f"{n_params_no_grad} parameters (requires_grad = False)")
