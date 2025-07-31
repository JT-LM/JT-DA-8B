import re
import asyncio

def has_code(response: str) -> list:
    """Check if the response contains code blocks.

    Args:
        response (str): The text response to check

    Returns:
        list: List of code blocks found in the response
    """
    pattern = r"```python(?:[a-zA-Z]*)\n(.*?)```"
    return re.findall(pattern, response, re.DOTALL)


def extract_text_and_code(input_string):
    start_marker = "```python"
    end_marker = "```"
    
    start_index = input_string.find(start_marker)
    end_index = input_string.find(end_marker, start_index + len(start_marker))
    if start_index != -1 and end_index != -1:
        result = input_string[:end_index + len(end_marker)]
    else:
        result = input_string
    
    return result

def code_run_result_parse(res_list):
    res_str = ''
    contain_img = False
    for item in res_list:
        if item['type'] == 'image_url' or item['type'] == 'file':
            contain_img = True
            break

    for item in res_list:
        if item['type'] == 'text' and not contain_img:
            res_str += f"{item['text']}"
        elif item['type'] == 'image_url':
            res_str += f"Figure has been generated: \n{item['image_url']}\n"
        elif item['type'] == 'file':
            res_str += f"File has been generated: \n{item['file_url']}\n"
    return res_str


async def stream_output(text, delay=0.05, chunk_size=3):
    for i in range(0, len(text), chunk_size):
        print(text[i:i+chunk_size], end='', flush=True)
        await asyncio.sleep(delay)

def code_print_change(input_string):
    
    lines = input_string.split("\n") 
    new_lines = []          

    for line in lines:
        marker = r"\.render" 
        pattern = r"^(\s*)" 
        positions = [match.start() for match in re.finditer(marker, line)]
        new_lines.append(line)
        match_s = re.match(pattern, line)
        if match_s:
            indent = match_s.group(1) 
        
        for pos in positions:
            stack = []
            index = 0
            sub_str = line[pos + len(marker) - 1:]
            
            if not sub_str or sub_str[index] != '(':
                continue
            
            stack.append('(')
            index += 1
            while index < len(sub_str) and stack:
                current_char = sub_str[index]
                
                if current_char == ')':
                    stack.pop()
                    if not stack: 
                        content = sub_str[1:index]  
                        if '=' in content:
                            content = content.split('=')[1]
                        print_code = f"{indent}print('html path:', {content})"
                        new_lines.append(print_code)
                        break
                        
                elif current_char == '(':
                    stack.append('(')
                index += 1

    return "\n".join(new_lines)

def process_and_remove_html_paths(text):
    lines = text.split("\n")  
    html_paths = []
    remaining_lines = []     
    for line in lines:
        line = line.strip() 
        if line.startswith("html path: ") and line.endswith(".html"):
            html_paths.append(line[len("html path: "):]) 
        else:
            remaining_lines.append(line) 
    return html_paths, "\n".join(remaining_lines)

def html_path_parse(res_list):
    html_paths_all = []
    for item in res_list:
        if item['type'] == 'text':
            html_paths, item['text'] = process_and_remove_html_paths(item['text'])
            html_paths_all.extend(html_paths)
    return html_paths_all, res_list
