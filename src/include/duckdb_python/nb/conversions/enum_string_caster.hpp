#pragma once

#include <nanobind/nanobind.h>
#include <cassert>
#include <cstdint>
#include <string>

//===----------------------------------------------------------------------===//
// Reusable nanobind type_caster macros for "string / integer or enum" arguments
//===----------------------------------------------------------------------===//
//
// Several DuckDB enums are registered as Python types via nb::enum_ AND given this caster, so a binding
// parameter typed as the enum also accepts a string (and, for most, an integer) naming one of its values.
// The caster handles three inputs: a str, an int, or a registered enum instance (read via its .value).
//
// The macros collapse the boilerplate into one invocation per enum, so the caster
// rewrite is a single-place change. nanobind requires from_python()/from_cpp() to be
// noexcept, so the DuckDB *FromString/*FromInteger calls (which throw on bad input)
// are wrapped: a bad value reports a generic conversion failure rather than the
// original InvalidInputException message (acceptable; refine post-cutover if needed).
//
// Invoke at GLOBAL scope (outside any namespace); each expands to a full
// `namespace nanobind { namespace detail { ... } }` specialization. Pass fully
// qualified names for the conversion functions and the enum type.

//! str + int + enum form.
#define DUCKDB_PY_ENUM_STRING_INT_CASTER(EnumType, FromStringFn, FromIntegerFn, NameLiteral)                           \
	namespace nanobind {                                                                                               \
	namespace detail {                                                                                                 \
	template <>                                                                                                        \
	struct type_caster<EnumType> {                                                                                     \
		NB_TYPE_CASTER(EnumType, const_name(NameLiteral))                                                              \
		bool from_python(handle src, uint8_t flags, cleanup_list *) noexcept {                                         \
			/* A registered enum instance is an EXACT match and is always accepted. str/int are lossy  */              \
			/* CONVERSIONS: gate them on cast_flags::convert so the no-convert overload pass can't      */             \
			/* mis-dispatch (matches nanobind's own enum caster).                                       */             \
			const bool convert = (flags & (uint8_t)nanobind::detail::cast_flags::convert) != 0;                        \
			try {                                                                                                      \
				/* Registered nb::enum_ instances aren't int subclasses, so accept a member  */                        \
				/* of the registered enum by reading its integer .value.                                         */    \
				nanobind::handle enum_type = nanobind::type<EnumType>();                                               \
				if (enum_type.is_valid() && PyObject_IsInstance(src.ptr(), enum_type.ptr()) == 1) {                    \
					value = FromIntegerFn(nanobind::cast<int64_t>(src.attr("value")));                                 \
					return true;                                                                                       \
				}                                                                                                      \
				if (convert && nanobind::isinstance<nanobind::str>(src)) {                                             \
					value = FromStringFn(nanobind::cast<std::string>(src));                                            \
					return true;                                                                                       \
				}                                                                                                      \
				if (convert && nanobind::isinstance<nanobind::int_>(src)) {                                            \
					value = FromIntegerFn(nanobind::cast<int64_t>(src));                                               \
					return true;                                                                                       \
				}                                                                                                      \
			} catch (...) {                                                                                            \
				return false;                                                                                          \
			}                                                                                                          \
			return false;                                                                                              \
		}                                                                                                              \
		static handle from_cpp(EnumType src, rv_policy, cleanup_list *) noexcept {                                     \
			/* Return the registered nb::enum_ member (not a bare int) so a function default renders as  */            \
			/* `Enum.MEMBER` in help()/stubs. Fall back to a bare int only if the enum type isn't        */            \
			/* registered yet (e.g. a default materialized before the enum bind ran).                    */            \
			nanobind::handle enum_type = nanobind::type<EnumType>();                                                   \
			/* N1: this default is materialized at bind time, so the enum's nb::enum_ registration must  */            \
			/* run first; a reorder makes type<EnumType>() invalid and silently falls back to a bare int  */           \
			/* (re-introducing #3). The assert makes that loud in debug; release no-ops + degrades below.  */          \
			assert(enum_type.is_valid() && "enum type must be registered before its default (finding #3/N1)");         \
			if (enum_type.is_valid()) {                                                                                \
				try {                                                                                                  \
					return enum_type(nanobind::int_((int64_t)src)).release();                                          \
				} catch (...) {                                                                                        \
				}                                                                                                      \
			}                                                                                                          \
			return nanobind::int_((int64_t)src).release();                                                             \
		}                                                                                                              \
	};                                                                                                                 \
	} /* namespace detail */                                                                                           \
	} /* namespace nanobind */

//! str + enum form (no integer accepted).
#define DUCKDB_PY_ENUM_STRING_CASTER(EnumType, FromStringFn, NameLiteral)                                              \
	namespace nanobind {                                                                                               \
	namespace detail {                                                                                                 \
	template <>                                                                                                        \
	struct type_caster<EnumType> {                                                                                     \
		NB_TYPE_CASTER(EnumType, const_name(NameLiteral))                                                              \
		bool from_python(handle src, uint8_t flags, cleanup_list *) noexcept {                                         \
			/* Exact registered-enum match is always accepted; the str CONVERSION is gated on          */              \
			/* cast_flags::convert so the no-convert overload pass can't mis-dispatch.                 */              \
			const bool convert = (flags & (uint8_t)nanobind::detail::cast_flags::convert) != 0;                        \
			try {                                                                                                      \
				/* Registered nb::enum_ instances aren't int subclasses; accept a member of the registered enum  */    \
				/* by reading its integer .value (this enum has no FromInteger, so cast the int directly).        */   \
				nanobind::handle enum_type = nanobind::type<EnumType>();                                               \
				if (enum_type.is_valid() && PyObject_IsInstance(src.ptr(), enum_type.ptr()) == 1) {                    \
					value = (EnumType)nanobind::cast<int64_t>(src.attr("value"));                                      \
					return true;                                                                                       \
				}                                                                                                      \
				if (convert && nanobind::isinstance<nanobind::str>(src)) {                                             \
					value = FromStringFn(nanobind::cast<std::string>(src));                                            \
					return true;                                                                                       \
				}                                                                                                      \
			} catch (...) {                                                                                            \
				return false;                                                                                          \
			}                                                                                                          \
			return false;                                                                                              \
		}                                                                                                              \
		static handle from_cpp(EnumType src, rv_policy, cleanup_list *) noexcept {                                     \
			/* Return the registered nb::enum_ member so defaults render as `Enum.MEMBER` in help()/stubs;   */        \
			/* fall back to a bare int if the enum type isn't registered yet.                                */        \
			nanobind::handle enum_type = nanobind::type<EnumType>();                                                   \
			/* N1: this default is materialized at bind time, so the enum's nb::enum_ registration must  */            \
			/* run first; a reorder makes type<EnumType>() invalid and silently falls back to a bare int  */           \
			/* (re-introducing #3). The assert makes that loud in debug; release no-ops + degrades below.  */          \
			assert(enum_type.is_valid() && "enum type must be registered before its default (finding #3/N1)");         \
			if (enum_type.is_valid()) {                                                                                \
				try {                                                                                                  \
					return enum_type(nanobind::int_((int64_t)src)).release();                                          \
				} catch (...) {                                                                                        \
				}                                                                                                      \
			}                                                                                                          \
			return nanobind::int_((int64_t)src).release();                                                             \
		}                                                                                                              \
	};                                                                                                                 \
	} /* namespace detail */                                                                                           \
	} /* namespace nanobind */
