// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <thread>
#include <chrono>

#include <pybind11/embed.h>
namespace py = pybind11;

int main() {
	try {
		py::scoped_interpreter guard {};

		py::exec(R"(
import asyncio
async def fast_coro(value, delay):
    await asyncio.sleep(delay)
    return value
)");

		auto coro = py::module::import("__main__").attr("fast_coro")(py::int_(42), py::float_(0.01));

		std::cout << "Main thread: created coroutine, spawning worker thread...\n";

		std::thread worker([coro]() mutable {
			try {
				py::gil_scoped_acquire gil;
				std::cerr << "Worker: acquired GIL\n";
				py::module asyncio = py::module::import("asyncio");
				py::object loop = asyncio.attr("new_event_loop")();
				asyncio.attr("set_event_loop")(loop);
				py::object task = asyncio.attr("ensure_future")(coro, py::arg("loop") = loop);
				std::cerr << "Worker: running loop.run_until_complete\n";
				py::object res = loop.attr("run_until_complete")(task);
				std::cerr << "Worker: run_until_complete returned" << std::endl;
				try {
					loop.attr("stop")();
				} catch (...) {
				}
				try {
					loop.attr("close")();
				} catch (...) {
				}
				std::cout << "Worker: result=" << py::str(res).cast<std::string>() << std::endl;
			} catch (const py::error_already_set &e) {
				std::cerr << "Worker: python exception: " << e.what() << std::endl;
				try {
					PyErr_Print();
				} catch (...) {
				}
			} catch (const std::exception &e) {
				std::cerr << "Worker: std::exception: " << e.what() << std::endl;
			} catch (...) {
				std::cerr << "Worker: unknown exception\n";
			}
		});

		// Wait for worker to run
		worker.join();

		std::cout << "Done\n";

	} catch (const std::exception &e) {
		std::cerr << "Main: exception: " << e.what() << std::endl;
		return 2;
	} catch (...) {
		std::cerr << "Main: unknown exception" << std::endl;
		return 3;
	}
	return 0;
}
