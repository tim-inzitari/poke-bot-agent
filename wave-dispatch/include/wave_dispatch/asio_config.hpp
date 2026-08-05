#pragma once

/** Include this before any <asio.hpp> in translation units. */
#if defined(WAVE_DISPATCH_HAS_IO_URING)
#ifndef ASIO_HAS_IO_URING
#define ASIO_HAS_IO_URING 1
#endif
#endif
