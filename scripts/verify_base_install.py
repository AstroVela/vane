"""Smoke-test the public Quickstart against a base Vane installation."""

import ray

import vane


def main() -> None:
    con = vane.connect()
    documents = con.values(  # noqa: F841 - resolved by DuckDB's Python replacement scan
        (
            vane.lit("doc-001").alias("document_id"),
            vane.lit("Signed claim form received.").alias("source_text"),
        ),
        (vane.lit("doc-002"), vane.lit("Invoice is missing approval metadata.")),
        (vane.lit("doc-003"), vane.lit("Policy reference has expired.")),
    )

    def review_status(source_text: object) -> str:
        text = str(source_text).lower()
        if "missing" in text or "expired" in text:
            return "needs_review"
        return "ready"

    vane.attach_function(
        review_status,
        alias="review_status_sql",
        connection=con,
        parameters=["VARCHAR"],
        return_dtype="VARCHAR",
    )

    try:
        reviewed = con.sql("""
            SELECT
                document_id,
                source_text,
                review_status_sql(source_text) AS review_status
            FROM documents
            ORDER BY document_id
        """)
        reviewed.show()
    finally:
        vane.detach_function("review_status_sql", connection=con)
        con.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
