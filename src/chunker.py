"""
Markdown-aware parent-child chunking
Respects headings, code blocks, paragraphs
"""
import json
import os
import re       #regular expressions for pattern matching

def extract_markdown_sections(text):
    """
    Split by markdown headings (##, ###, etc.)
    Keep code blocks intact
    """
    # Find all headings
    heading_pattern = r'^(#{1,6})\s+(.+)$'          #match r with ^ new line, #1-6 hashtags, \s+ one or more whitespace characters, (.+) everything after the #, $ for end of line
    sections = []       #list to store all sections found
    current_section = {"title": "Introduction", "content": ""}

    for line in text.split('\n'):
        heading_match = re.match(heading_pattern, line, re.MULTILINE)           #check if line is heading

        if heading_match:
            # Save previous section
            if current_section["content"].strip():
                sections.append(current_section)

            # Start new section
            current_section = {
                "title": heading_match.group(2),
                "content": line + "\n"
            }
        else:
            current_section["content"] += line + "\n"

    # Don't forget last section
    if current_section["content"].strip():
        sections.append(current_section)

    return sections

def split_section_smart(section_text, max_words=300):
    """
    Split a section, but keep code blocks intact
    """
    chunks = []

    # Detect code blocks
    code_block_pattern = r'```[\s\S]*?```'

    # Split into code and non-code parts
    parts = re.split(f'({code_block_pattern})', section_text)

    current_chunk = ""

    for part in parts:
        is_code = part.startswith('```')

        if is_code:
            # Code block - keep intact
            if len(current_chunk.split()) > 0:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(part.strip())
        else:
            # Regular text - can split
            words = part.split()
            for word in words:
                current_chunk += word + " "
                if len(current_chunk.split()) >= max_words:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def chunk_parent_child_markdown_aware(documents):
    """
    Parent = full section (by heading)
    Children = smart splits within section
    """
    all_chunks = []
    chunk_id = 0

    for doc in documents:
        # Extract sections by headings
        sections = extract_markdown_sections(doc['content'])

        for section in sections:
            # Parent = full section
            parent_text = section['content']
            parent_id = f"parent_{chunk_id}"

            # Children = smart splits
            child_texts = split_section_smart(section['content'], max_words=300)

            children = []
            for i, child_text in enumerate(child_texts):
                child_id = f"child_{chunk_id}_{i}"

                # Detect if this chunk has code
                has_code = '```' in child_text

                children.append({
                    'id': child_id,
                    'text': child_text,
                    'parent_id': parent_id,
                    'metadata': {
                        'section_title': section['title'],
                        'has_code': has_code,
                        'source_url': doc['url']
                    }
                })

            all_chunks.append({
                'parent': {
                    'id': parent_id,
                    'text': parent_text,
                    'url': doc['url'],
                    'title': section['title']
                },
                'children': children
            })

            chunk_id += 1

    return all_chunks

if __name__ == "__main__":
    with open('data/raw/docs.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)

    chunks = chunk_parent_child_markdown_aware(documents)

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    total_children = sum(len(c['children']) for c in chunks)
    children_with_code = sum(
        1 for c in chunks
        for child in c['children']
        if child['metadata']['has_code']
    )

    print(f"\n✓ Created {len(chunks)} parents")
    print(f"✓ Created {total_children} children")
    print(f"✓ {children_with_code} children contain code blocks")
    print(f"✓ Saved to data/processed/chunks.json")