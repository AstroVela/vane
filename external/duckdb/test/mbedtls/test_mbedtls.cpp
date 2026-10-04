#include "catch.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "mbedtls_wrapper.hpp"

#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>

using namespace duckdb_mbedtls;
using namespace std;

static string file_to_string(string filename) {
	std::ifstream stream(filename, ios_base::binary);
	duckdb::stringstream buffer;
	buffer << stream.rdbuf();
	return buffer.str();
}

TEST_CASE("Test that we can verify a signature", "[mbedtls]") {
	// those files are created with the create_files.sh script
	auto file_content = file_to_string("test/mbedtls/dummy_file");
	auto signature = file_to_string("test/mbedtls/dummy_file.signature");
	auto pubkey = file_to_string("test/mbedtls/public.pem");

	auto hash = MbedTlsWrapper::ComputeSha256Hash(file_content);
	REQUIRE(MbedTlsWrapper::IsValidSha256Signature(pubkey, signature, hash));
	string empty_string = "";

	auto borked_pubkey = pubkey;
	borked_pubkey[10]++;

	// a borked public key is an exception, this should never happen
	REQUIRE_THROWS(MbedTlsWrapper::IsValidSha256Signature(borked_pubkey, signature, hash));
	REQUIRE_THROWS(MbedTlsWrapper::IsValidSha256Signature(empty_string, signature, hash));

	// wrong-length signatures or hashes should never happen either
	REQUIRE_THROWS(MbedTlsWrapper::IsValidSha256Signature(pubkey, empty_string, hash));
	REQUIRE_THROWS(MbedTlsWrapper::IsValidSha256Signature(pubkey, signature, empty_string));

	// lets flip some bits in the file, it should not validate
	auto borked_file = file_content;
	borked_file[10]++;
	auto hash2 = MbedTlsWrapper::ComputeSha256Hash(borked_file);
	REQUIRE(!MbedTlsWrapper::IsValidSha256Signature(pubkey, signature, hash2));

	auto borked_signature = signature;
	borked_signature[10]++;
	REQUIRE(!MbedTlsWrapper::IsValidSha256Signature(pubkey, borked_signature, hash));

	auto borked_hash = hash;
	borked_hash[10]++;
	auto hash3 = MbedTlsWrapper::ComputeSha256Hash(empty_string);
	REQUIRE(!MbedTlsWrapper::IsValidSha256Signature(pubkey, signature, hash3));
	REQUIRE(!MbedTlsWrapper::IsValidSha256Signature(pubkey, signature, borked_hash));

	// seems all right!
	REQUIRE(MbedTlsWrapper::IsValidSha256Signature(pubkey, signature, hash));
}

TEST_CASE("Vane production extension signer is trusted without community keys", "[mbedtls][vane-signing]") {
	// Public verification fixture, not an extension artifact. The production
	// private key is independently managed and is never needed to run this test.
	const string message = "Vane production extension trust regression fixture. This is not an extension artifact.\n";
	const unsigned char signature_bytes[] = {
	    0x47, 0xf5, 0xdf, 0x1a, 0x2e, 0xf8, 0x60, 0x50, 0xed, 0xd8, 0x94, 0xd9, 0xde, 0xba, 0x94, 0x8a, 0xcd, 0xd1,
	    0x4c, 0x40, 0xc1, 0x5f, 0x1c, 0x9f, 0xe5, 0xc3, 0xa8, 0x51, 0x54, 0xe4, 0x43, 0x86, 0xfe, 0xd1, 0xa0, 0xd8,
	    0xb5, 0x28, 0x8b, 0x50, 0x59, 0xb0, 0x2e, 0xea, 0xfd, 0x75, 0x40, 0xde, 0xb4, 0xf4, 0x93, 0xe4, 0x86, 0x9d,
	    0x6b, 0x50, 0x4f, 0xaa, 0x86, 0x66, 0x71, 0x27, 0x20, 0x7d, 0x7b, 0xf6, 0x4f, 0x26, 0x52, 0x85, 0x67, 0x9b,
	    0x7c, 0x69, 0x0f, 0xea, 0x2a, 0xab, 0x1f, 0xf0, 0x01, 0xc1, 0x2e, 0x4c, 0xbf, 0x56, 0x9c, 0x01, 0x95, 0x4b,
	    0x69, 0x27, 0x3f, 0x46, 0xe5, 0x14, 0xb7, 0xd7, 0x89, 0xec, 0xd1, 0x52, 0x70, 0xcc, 0x72, 0x19, 0xae, 0xf2,
	    0x43, 0x5e, 0xa3, 0x30, 0x8c, 0xa1, 0xd0, 0x34, 0x96, 0xec, 0x4d, 0x4e, 0x6a, 0x0f, 0x01, 0xb3, 0xda, 0xf0,
	    0xe2, 0xe6, 0x83, 0x6d, 0x7d, 0xeb, 0xb4, 0x1e, 0x71, 0x02, 0xa0, 0x0c, 0x7c, 0xaa, 0xdc, 0xac, 0xfc, 0x16,
	    0x34, 0x23, 0x75, 0x02, 0xa6, 0x30, 0xf7, 0x03, 0x7a, 0x1a, 0x3f, 0x3d, 0x95, 0x1c, 0xab, 0x96, 0x9c, 0xf4,
	    0xce, 0x52, 0x9f, 0xf1, 0x9a, 0x66, 0xc0, 0x15, 0x94, 0x96, 0x8d, 0xe3, 0xf6, 0x0e, 0x92, 0x22, 0xde, 0xc2,
	    0x55, 0xfa, 0x14, 0xae, 0xa0, 0x8a, 0xb3, 0xfc, 0xfe, 0xd7, 0xeb, 0xbc, 0xa7, 0x88, 0x62, 0x3d, 0x75, 0xcf,
	    0x1f, 0xc1, 0xad, 0xb3, 0xc8, 0x8c, 0xd1, 0xc0, 0x62, 0xed, 0x93, 0x2f, 0xaf, 0xe5, 0x89, 0x3a, 0x48, 0xeb,
	    0x18, 0x29, 0x40, 0x4d, 0xef, 0x31, 0xaf, 0xc3, 0x98, 0xd2, 0x7c, 0xf8, 0x2f, 0x10, 0x0e, 0x28, 0x53, 0x1d,
	    0xa0, 0xd7, 0x3d, 0xa2, 0x46, 0x21, 0x47, 0x49, 0xaf, 0xda, 0x85, 0xe5, 0x76, 0x71, 0x37, 0x95, 0x09, 0x56,
	    0x92, 0xde, 0x10, 0xf3};
	const string signature(reinterpret_cast<const char *>(signature_bytes), sizeof(signature_bytes));
	const auto hash = MbedTlsWrapper::ComputeSha256Hash(message);
	string production_key;
	unsigned matching_keys = 0;
	for (const auto &key : duckdb::ExtensionHelper::GetPublicKeys(false)) {
		if (MbedTlsWrapper::IsValidSha256Signature(key, signature, hash)) {
			production_key = key;
			matching_keys++;
		}
	}
	REQUIRE(matching_keys == 1);
	REQUIRE(!MbedTlsWrapper::IsValidSha256Signature(production_key, signature,
	                                                MbedTlsWrapper::ComputeSha256Hash(message + "changed")));
	auto changed_signature = signature;
	changed_signature[0] ^= 1;
	REQUIRE(!MbedTlsWrapper::IsValidSha256Signature(production_key, changed_signature, hash));

	// The production public key cannot validate the committed CI-test signer.
	const auto ci_hash = MbedTlsWrapper::ComputeSha256Hash(file_to_string("test/mbedtls/dummy_file"));
	const auto ci_signature = file_to_string("test/mbedtls/dummy_file.signature");
	REQUIRE(!MbedTlsWrapper::IsValidSha256Signature(production_key, ci_signature, ci_hash));
}
