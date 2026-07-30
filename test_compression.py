import unittest
import uuid
from pathlib import Path

from compress import (
    FILE_MAGIC,
    FILE_VERSION,
    HEADER_STRUCT,
    write_compressed_file,
)
from huffman_encoding import (
    build_huffman_codebook,
    encode_integers,
)


class HuffmanEncodingTests(unittest.TestCase):
    def test_positive_negative_and_zero_can_be_encoded(self) -> None:
        frequencies = {-3: 2, 0: 5, 4: 3}
        codebook = build_huffman_codebook(frequencies)

        encoded_data, valid_bit_count = encode_integers(
            [-3, 0, 4, 0],
            codebook,
        )

        self.assertIsInstance(encoded_data, bytes)
        self.assertGreater(valid_bit_count, 0)

    def test_last_byte_is_padded_without_counting_padding(self) -> None:
        codebook = {-1: "0", 0: "10", 1: "11"}

        encoded_data, valid_bit_count = encode_integers(
            [-1, 0, 1],
            codebook,
        )

        # 位串是 01011，共 5 个有效位；右侧补三个零后得到 01011000。
        self.assertEqual(valid_bit_count, 5)
        self.assertEqual(encoded_data, bytes([0b01011000]))

    def test_one_symbol_gets_a_nonempty_code(self) -> None:
        codebook = build_huffman_codebook({7: 4})
        encoded_data, valid_bit_count = encode_integers(
            [7, 7, 7, 7],
            codebook,
        )

        self.assertEqual(codebook, {7: "0"})
        self.assertEqual(valid_bit_count, 4)
        self.assertEqual(encoded_data, bytes([0]))

    def test_empty_input_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "整数序列为空"):
            encode_integers([], {0: "0"})

    def test_unknown_integer_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "不在霍夫曼编码表中"):
            encode_integers([0, 9], {0: "0"})

    def test_output_is_binary_bytes_not_zero_one_text(self) -> None:
        encoded_data, _ = encode_integers(
            [-1, 0, 1],
            {-1: "0", 0: "10", 1: "11"},
        )

        self.assertIsInstance(encoded_data, bytes)
        self.assertEqual(encoded_data, b"\x58")
        self.assertNotEqual(encoded_data, b"01011000")

    def test_same_frequencies_create_the_same_codebook(self) -> None:
        first = build_huffman_codebook({2: 4, -1: 4, 0: 2})
        second = build_huffman_codebook({0: 2, 2: 4, -1: 4})
        self.assertEqual(first, second)

    def test_more_frequent_symbol_has_no_longer_code(self) -> None:
        codebook = build_huffman_codebook({0: 20, 1: 5, 2: 1})
        self.assertLessEqual(len(codebook[0]), len(codebook[1]))
        self.assertLessEqual(len(codebook[1]), len(codebook[2]))


class CompressedFileTests(unittest.TestCase):
    def test_file_size_and_header_element_count(self) -> None:
        huffman_data, valid_bit_count = encode_integers(
            [0, 1, 0, 1],
            {0: "0", 1: "1"},
        )

        output_path = (
            Path.cwd() / f".compression_test_{uuid.uuid4().hex}.bin"
        )
        try:
            header_size = write_compressed_file(
                output_path=output_path,
                latent_shape=(1, 2, 2),
                element_count=4,
                valid_bit_count=valid_bit_count,
                huffman_data=huffman_data,
            )

            file_data = output_path.read_bytes()
            self.assertEqual(
                len(file_data),
                header_size + len(huffman_data),
            )

            header_values = HEADER_STRUCT.unpack(
                file_data[:HEADER_STRUCT.size]
            )
            (
                magic,
                version,
                channels,
                height,
                width,
                element_count,
                header_valid_bit_count,
            ) = header_values

            self.assertEqual(magic, FILE_MAGIC)
            self.assertEqual(version, FILE_VERSION)
            self.assertEqual((channels, height, width), (1, 2, 2))
            self.assertEqual(element_count, 4)
            self.assertEqual(header_valid_bit_count, valid_bit_count)
        finally:
            output_path.unlink(missing_ok=True)

    def test_existing_output_file_is_not_overwritten(self) -> None:
        output_path = (
            Path.cwd() / f".compression_test_{uuid.uuid4().hex}.bin"
        )
        try:
            original_file_data = b"original file"
            output_path.write_bytes(original_file_data)

            with self.assertRaisesRegex(FileExistsError, "请更换"):
                write_compressed_file(
                    output_path=output_path,
                    latent_shape=(1, 1, 1),
                    element_count=1,
                    valid_bit_count=1,
                    huffman_data=b"\x00",
                )

            self.assertEqual(output_path.read_bytes(), original_file_data)
        finally:
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
