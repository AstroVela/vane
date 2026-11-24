// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "module_bindings.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pybind11/conversions/pyconnection_default.hpp"
#include "vane_python/pyrelation.hpp"
#include "vane_python/pyresult.hpp"

namespace duckdb {

void InitializeModuleResultsBindings(py::module_ &m) {
	m.def(
	    "fetchone",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchOne();
	    },
	    "Fetch a single row from a result following execute", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "fetchmany",
	    [](idx_t size, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchMany(size);
	    },
	    "Fetch the next set of rows from a result following execute", py::arg("size") = 1, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "fetchall",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchAll();
	    },
	    "Fetch all rows from a result following execute", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "fetchnumpy",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchNumpy();
	    },
	    "Fetch a result as list of NumPy arrays following execute", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "fetchdf",
	    [](bool date_as_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", py::kw_only(), py::arg("date_as_object") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "fetch_df",
	    [](bool date_as_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", py::kw_only(), py::arg("date_as_object") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "df",
	    [](bool date_as_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", py::kw_only(), py::arg("date_as_object") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "fetch_df_chunk",
	    [](const idx_t vectors_per_chunk = 1, bool date_as_object = false,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDFChunk(vectors_per_chunk, date_as_object);
	    },
	    "Fetch a chunk of the result as DataFrame following execute()", py::arg("vectors_per_chunk") = 1, py::kw_only(),
	    py::arg("date_as_object") = false, py::arg("connection") = py::none());
	m.def(
	    "pl",
	    [](idx_t rows_per_batch, bool lazy, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchPolars(rows_per_batch, lazy);
	    },
	    "Fetch a result as Polars DataFrame following execute()", py::arg("rows_per_batch") = 1000000, py::kw_only(),
	    py::arg("lazy") = false, py::arg("connection") = py::none());
	m.def(
	    "to_arrow_table",
	    [](idx_t batch_size, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchArrow(batch_size);
	    },
	    "Fetch a result as Arrow table following execute()", py::arg("batch_size") = 1000000, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "to_arrow_reader",
	    [](idx_t batch_size, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchRecordBatchReader(batch_size);
	    },
	    "Fetch an Arrow RecordBatchReader following execute()", py::arg("batch_size") = 1000000, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "fetch_arrow_table",
	    [](idx_t rows_per_batch, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    PyErr_WarnEx(PyExc_DeprecationWarning, "fetch_arrow_table() is deprecated, use to_arrow_table() instead.",
		                 0);
		    return conn->FetchArrow(rows_per_batch);
	    },
	    "Fetch a result as Arrow table following execute()", py::arg("rows_per_batch") = 1000000, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "fetch_record_batch",
	    [](const idx_t rows_per_batch, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    PyErr_WarnEx(PyExc_DeprecationWarning, "fetch_record_batch() is deprecated, use to_arrow_reader() instead.",
		                 0);
		    return conn->FetchRecordBatchReader(rows_per_batch);
	    },
	    "Fetch an Arrow RecordBatchReader following execute()", py::arg("rows_per_batch") = 1000000, py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "torch",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchPyTorch();
	    },
	    "Fetch a result as dict of PyTorch Tensors following execute()", py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "tf",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchTF();
	    },
	    "Fetch a result as dict of TensorFlow Tensors following execute()", py::kw_only(),
	    py::arg("connection") = py::none());
}

void InitializeModuleResultPropertiesBindings(py::module_ &m) {
	m.def(
	    "description",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetDescription();
	    },
	    "Get result set attributes, mainly column names", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "rowcount",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetRowcount();
	    },
	    "Get result set row count", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "arrow",
	    [](idx_t rows_per_batch, shared_ptr<DuckDBPyConnection> conn) -> duckdb::pyarrow::RecordBatchReader {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchRecordBatchReader(rows_per_batch);
	    },
	    "Alias of to_arrow_reader(). We recommend using to_arrow_reader() instead.",
	    py::arg("rows_per_batch") = 1000000, py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "arrow",
	    [](py::object &arrow_object, shared_ptr<DuckDBPyConnection> conn) -> unique_ptr<DuckDBPyRelation> {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromArrow(arrow_object);
	    },
	    "Create a relation object from an Arrow object", py::arg("arrow_object"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "df",
	    [](bool date_as_object, shared_ptr<DuckDBPyConnection> conn) -> PandasDataFrame {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FetchDF(date_as_object);
	    },
	    "Fetch a result as DataFrame following execute()", py::kw_only(), py::arg("date_as_object") = false,
	    py::arg("connection") = py::none());
	m.def(
	    "df",
	    [](const PandasDataFrame &value, shared_ptr<DuckDBPyConnection> conn) -> unique_ptr<DuckDBPyRelation> {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(value);
	    },
	    "Create a relation object from the DataFrame df", py::arg("df"), py::kw_only(),
	    py::arg("connection") = py::none());
}

} // namespace duckdb
