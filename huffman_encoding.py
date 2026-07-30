import heapq
from collections.abc import Iterable, Mapping


class _HuffmanNode:
    """霍夫曼树中的一个节点，仅供本文件内部使用。"""

    def __init__(
        self,
        symbol: int | None = None,
        left: "_HuffmanNode | None" = None,
        right: "_HuffmanNode | None" = None,
    ) -> None:
        self.symbol = symbol
        self.left = left
        self.right = right


def build_huffman_codebook(
    frequencies: Mapping[int, int],
) -> dict[int, str]:
    """根据整数出现次数建立稳定的霍夫曼编码表。"""
    if not frequencies:
        raise ValueError("整数频数为空，无法建立霍夫曼编码表。")

    # 堆中的排序字段依次是：出现次数、子树中的最小整数、创建顺序。
    # 这些字段会消除出现次数相同时的不确定性，保证相同统计数据总能
    # 生成相同的编码表。
    heap: list[tuple[int, int, int, _HuffmanNode]] = []
    creation_order = 0

    for symbol in sorted(frequencies):
        count = frequencies[symbol]
        if not isinstance(symbol, int):
            raise TypeError("霍夫曼编码表中的符号必须是整数。")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"整数 {symbol} 的出现次数必须是大于 0 的整数。"
            )

        node = _HuffmanNode(symbol=symbol)
        heapq.heappush(
            heap,
            (count, symbol, creation_order, node),
        )
        creation_order += 1

    # 只有一种整数时也必须给它一个非空编码，否则无法记录元素数量。
    if len(heap) == 1:
        only_symbol = heap[0][3].symbol
        if only_symbol is None:
            raise RuntimeError("建立单符号霍夫曼编码表时发生内部错误。")
        return {only_symbol: "0"}

    while len(heap) > 1:
        left_count, left_min_symbol, _, left_node = heapq.heappop(heap)
        right_count, right_min_symbol, _, right_node = heapq.heappop(heap)

        parent_node = _HuffmanNode(
            left=left_node,
            right=right_node,
        )
        parent_count = left_count + right_count
        parent_min_symbol = min(left_min_symbol, right_min_symbol)
        heapq.heappush(
            heap,
            (
                parent_count,
                parent_min_symbol,
                creation_order,
                parent_node,
            ),
        )
        creation_order += 1

    root = heap[0][3]
    codebook: dict[int, str] = {}

    def visit(node: _HuffmanNode, prefix: str) -> None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
            return

        if node.left is None or node.right is None:
            raise RuntimeError("霍夫曼树结构不完整。")

        visit(node.left, prefix + "0")
        visit(node.right, prefix + "1")

    visit(root, "")
    return codebook


def encode_integers(
    integer_values: Iterable[int],
    codebook: Mapping[int, str],
) -> tuple[bytes, int]:
    """把整数序列编码成真正的字节，并返回实际有效位数。"""
    if not codebook:
        raise ValueError("霍夫曼编码表为空，无法编码整数序列。")

    encoded_bytes = bytearray()
    current_byte = 0
    bits_in_current_byte = 0
    valid_bit_count = 0
    value_count = 0

    for value in integer_values:
        value_count += 1
        if value not in codebook:
            raise ValueError(
                f"整数 {value} 不在霍夫曼编码表中，"
                "请重新运行 build_codebook.py。"
            )

        code = codebook[value]
        if not code or any(bit not in "01" for bit in code):
            raise ValueError(
                f"整数 {value} 对应的霍夫曼编码不是有效的二进制编码。"
            )

        # 每读到一个二进制位，就把它移入当前字节。凑满 8 位后，
        # 才向结果中写入一个真正的字节。
        for bit in code:
            current_byte = (current_byte << 1) | int(bit)
            bits_in_current_byte += 1
            valid_bit_count += 1

            if bits_in_current_byte == 8:
                encoded_bytes.append(current_byte)
                current_byte = 0
                bits_in_current_byte = 0

    if value_count == 0:
        raise ValueError("整数序列为空，没有可以进行霍夫曼编码的数据。")

    # 最后不足 8 位时在右侧补零。补零只用于组成完整字节，
    # 因此不会增加 valid_bit_count。
    if bits_in_current_byte > 0:
        current_byte <<= 8 - bits_in_current_byte
        encoded_bytes.append(current_byte)

    return bytes(encoded_bytes), valid_bit_count


def decode_integers(
    encoded_data: bytes,
    valid_bit_count: int,
    codebook: Mapping[int, str],
    expected_value_count: int,
) -> list[int]:
    """把霍夫曼字节数据恢复为整数序列。"""
    if not isinstance(encoded_data, bytes):
        raise TypeError("霍夫曼数据必须是 bytes 类型的字节数据。")
    if valid_bit_count <= 0:
        raise ValueError("霍夫曼数据的有效位数必须大于 0。")
    if expected_value_count <= 0:
        raise ValueError("期望解码出的整数数量必须大于 0。")
    if not codebook:
        raise ValueError("霍夫曼编码表为空，无法解码字节数据。")

    expected_byte_count = (valid_bit_count + 7) // 8
    if len(encoded_data) < expected_byte_count:
        raise ValueError(
            "霍夫曼数据提前结束：实际字节数不足以容纳文件头记录的有效位。"
        )
    if len(encoded_data) > expected_byte_count:
        raise ValueError(
            "霍夫曼数据长度错误：有效数据之后存在多余字节。"
        )

    # 编码时最后一个字节会在右侧补零。这里不仅忽略这些无效位，
    # 还检查补齐位置是否确实为零，以便尽早发现文件损坏。
    padding_bit_count = (8 - valid_bit_count % 8) % 8
    if padding_bit_count > 0:
        padding_mask = (1 << padding_bit_count) - 1
        if encoded_data[-1] & padding_mask:
            raise ValueError("霍夫曼数据末尾的补齐位不是零，文件可能已损坏。")

    # codebook 是“整数 -> 编码”，解码时需要建立“编码 -> 整数”的反向表。
    reverse_codebook: dict[str, int] = {}
    for integer_value, code in codebook.items():
        if not isinstance(integer_value, int):
            raise TypeError("霍夫曼编码表中的符号必须是整数。")
        if not isinstance(code, str) or not code:
            raise ValueError(
                f"整数 {integer_value} 对应的霍夫曼编码为空或类型错误。"
            )
        if any(bit not in "01" for bit in code):
            raise ValueError(
                f"整数 {integer_value} 对应的霍夫曼编码包含非二进制字符。"
            )
        if code in reverse_codebook:
            raise ValueError("霍夫曼编码表中存在重复编码，无法唯一解码。")
        reverse_codebook[code] = integer_value

    # 霍夫曼编码必须满足前缀规则：一个完整编码不能是另一个编码的前缀。
    valid_prefixes: set[str] = set()
    for code in reverse_codebook:
        for prefix_length in range(1, len(code)):
            valid_prefixes.add(code[:prefix_length])

    for prefix in valid_prefixes:
        if prefix in reverse_codebook:
            raise ValueError("霍夫曼编码表不满足前缀规则，无法唯一解码。")

    decoded_values: list[int] = []
    current_code = ""

    # 编码时每个字节都从最高位开始写入，所以解码也按相同顺序逐位读取。
    for bit_index in range(valid_bit_count):
        byte_value = encoded_data[bit_index // 8]
        bit_offset = 7 - bit_index % 8
        bit = (byte_value >> bit_offset) & 1
        current_code += str(bit)

        if current_code in reverse_codebook:
            decoded_values.append(reverse_codebook[current_code])
            current_code = ""

            if len(decoded_values) > expected_value_count:
                raise ValueError(
                    "解码出的整数数量超过文件头记录的元素总数，"
                    "压缩文件或编码表可能不匹配。"
                )
        elif current_code not in valid_prefixes:
            raise ValueError(
                f"霍夫曼数据在第 {bit_index + 1} 个有效位附近无法匹配，"
                "文件可能已损坏或编码表不正确。"
            )

    if current_code:
        raise ValueError(
            "霍夫曼数据在一个编码尚未结束时提前终止，文件可能已损坏。"
        )
    if len(decoded_values) != expected_value_count:
        raise ValueError(
            "解码出的整数数量与文件头记录的元素总数不一致："
            f"期望 {expected_value_count}，实际 {len(decoded_values)}。"
        )

    return decoded_values
