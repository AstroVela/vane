// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parser/parsed_expression.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/parser/base_expression.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/common/shared_ptr.hpp"

namespace duckdb {
class Deserializer;
class Serializer;

//! QueryRelation's first bind records names that depend on argument expansion.
//! This transient capture is shared by AST copies and is not serialized.
struct UnpackedColumnNameCapture {
	bool active = true;
	string name;
};

//!  The ParsedExpression class is a base class that can represent any expression
//!  part of a SQL statement.
/*!
 The ParsedExpression class is a base class that can represent any expression
 part of a SQL statement. This is, for example, a column reference in a SELECT
 clause, but also operators, aggregates or filters. The Expression is emitted by the parser and does not contain any
 information about bindings to the catalog or to the types. ParsedExpressions are transformed into regular Expressions
 in the Binder.
 */
class ParsedExpression : public BaseExpression {
public:
	//! Create an Expression
	ParsedExpression(ExpressionType type, ExpressionClass expression_class) : BaseExpression(type, expression_class) {
	}

public:
	shared_ptr<UnpackedColumnNameCapture> unpacked_column_name;

	bool IsAggregate() const override;
	bool IsWindow() const override;
	bool HasSubquery() const override;
	bool IsScalar() const override;
	bool HasParameter() const override;

	bool Equals(const BaseExpression &other) const override;
	hash_t Hash() const override;

	//! Create a copy of this expression
	virtual unique_ptr<ParsedExpression> Copy() const = 0;

	virtual void Serialize(Serializer &serializer) const;
	static unique_ptr<ParsedExpression> Deserialize(Deserializer &deserializer);

	static bool Equals(const unique_ptr<ParsedExpression> &left, const unique_ptr<ParsedExpression> &right);
	static bool ListEquals(const vector<unique_ptr<ParsedExpression>> &left,
	                       const vector<unique_ptr<ParsedExpression>> &right);

protected:
	//! Copy base Expression properties from another expression to this one,
	//! used in Copy method
	void CopyProperties(const ParsedExpression &other) {
		type = other.type;
		expression_class = other.expression_class;
		alias = other.alias;
		query_location = other.query_location;
		unpacked_column_name = other.unpacked_column_name;
	}
};

} // namespace duckdb
