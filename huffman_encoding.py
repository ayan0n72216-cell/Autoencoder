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
