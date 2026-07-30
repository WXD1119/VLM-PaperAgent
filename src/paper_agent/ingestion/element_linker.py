from paper_agent.domain.paper import PaperElement


def link_adjacent_elements(elements: list[PaperElement]) -> list[PaperElement]:
    """按文档阅读顺序设置确定性的双向链接。"""
    for index, element in enumerate(elements):
        element.prev_id = elements[index - 1].element_id if index > 0 else None
        element.next_id = elements[index + 1].element_id if index + 1 < len(elements) else None
    return elements
