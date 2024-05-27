import os
import re
import math
import logging


def load_a_file(a_file, content=None):
    if not content:
        content = []
    with open(a_file, 'r', encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        content.extend(lines)
    return content


def save_a_file(a_file_content, file_name):
    with open(file_name, "w", encoding='utf-8') as f:
        f.writelines(a_file_content)


def get_file_list(folder):
    filelist = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if file_type in file_type_list:
                file_fullname = os.path.join(dirpath, file)
                filelist.append(file_fullname)

    return filelist


def process_gen_files(gen_path_dir, len_ori):
    txt_files_lst = get_file_list(gen_path_dir)
    txt_files_lst = sorted(txt_files_lst, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    content = []
    for i in range(len(txt_files_lst)):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file, content)
    diff_len = len(content) - len_ori

    if diff_len > len(txt_files_lst) or diff_len < 0:
        return content

    content_all = []
    for i in range(diff_len):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file)
        content_all.extend(content[:-1])
    for i in range(diff_len, len(txt_files_lst)):
        a_file = txt_files_lst[i]
        content = load_a_file(a_file)
        content_all.extend(content)

    return content_all


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def replace_transmean(text):
    if not text:
        return
    if '\\' in text:
        text = text.replace("\\", "\\\\")
    pattern = r'([\[\]\(\)\{\}\.\*\+\?\^\-])'
    replacement = r'\\\1'
    replaced_text = re.sub(pattern, replacement, text)

    return replaced_text


def replace_fraction(text):
    if 'frac' not in text:
        return text
    f_t = re.findall(r'(?:\s*\}){3,}', text)
    if f_t:
        return text
    pattern = r'\\[d]?frac\{([\d,.\\pi]+)\}\{([\d,.]+)\}'
    replacement = r'{\1/\2}'
    replaced_text = re.sub(pattern, replacement, text)

    return replaced_text


def get_matched_string(text, start_index):
    stack = ['{']
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '{':
            stack.append('{')
        elif char == '}':
            end_index = i
            if len(stack) > 0:
                stack.pop()
                if len(stack) == 0:
                    break
            else:
                stack.append('}')
                break

    if len(stack) > 0:
        end_index = -1

    return end_index


def replace_frac(text):
    find_str = '\\frac{'

    while 'frac' in text:
        start_index = text.rfind(find_str) + len(find_str)
        end_index = get_matched_string(text, start_index)
        frac_a = text[start_index:end_index]
        end_index_b = get_matched_string(text, end_index+2)
        frac_b = text[end_index+2:end_index_b]
        if frac_a.isdigit() and frac_b.isdigit():
            text = text[:text.rfind(find_str)-1] + frac_a + '/' + frac_b
        else:
            text = text[:text.rfind(find_str)-1] + '{' + frac_a + '/' + frac_b + '}'

    return text


def has_numbers(input_string):
    pattern = r'\d+'
    match = re.search(pattern, input_string)
    return bool(match)


def process_content(qa_content, ans_true):
    if not qa_content:
        return None

    qa_content = re.sub('(\s*<n>\s*)+?$', '', qa_content)
    qa_content = re.sub(r'\\boxed{\s*([\d.,/]+)\s*}[.]?', '\\1', qa_content)
    q_text, a_content = qa_content.split('[SEP]')
    ans_all = a_content.strip().replace('<br>', '<n>').replace('\!', '').replace('\;', '').replace('\,', '').replace('$', '').replace('≈', '=').replace('</n>', '<n>')
    ans_all = re.sub(r'(<n>\s*)+', '<n>', ans_all)

    ans_all = ans_all.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "")
    ans_all = ans_all.replace('$$', ' ').replace('$', '').replace('≈', '=').replace('等于', '=')
    ans_all = ans_all.replace('，', ',').replace('\\(', '').replace('\\)', '').replace('\\]', '').replace('\\[', '')
    ans_all = ans_all.replace('\\cup', '∪').replace('\\cup', '∩').replace('<n>。', '')
    ans_all = re.sub(r'\.$', '', ans_all)
    ans_all = re.sub(r'(<n>)+$', '', ans_all)
    if 'sqrt' in ans_all:
        ans_all = re.sub(r'(?<![\\])sqrt', r'\\sqrt', ans_all)
    if 'sqrt' in ans_true:
        ans_true = _fix_sqrt(ans_true)

    ans_true = ans_true.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "").replace(' ', '').replace(
        "tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "").replace("\!", "").replace('\\mbox{inches}^2', '')
    ans_true_yuan = ans_true

    # fix frac
    if 'frac' in ans_true:
        try:
            ans_true = _fix_fracs(ans_true)
        except Exception as e:
            print('Error: ', e)

        ans_true_text = replace_fraction(ans_true)
    else:
        ans_true_text = ans_true

    frac_1 = re.findall(r'\d+\{\d+/\d+\}', ans_true_text)
    if frac_1 and frac_1[0] == ans_true_text:
        num_1 = frac_1[0].split('{')[0]
        num_2 = frac_1[0].split('{')[1].split('/')[0]
        num_3 = frac_1[0].split('/')[1].replace('}', '')
        num_str = str(int(num_1)*int(num_3)+int(num_2)) + '/' + num_3
        if num_str in ans_all:
            ans_true_text = num_str

    ans_true_text = re.sub(
        r'(?<![a-z\d^}])\{((?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+\_\d+)|(?:[-]?\d+))\}(?![\d])', '\\1', ans_true_text)
    ans_true_text = re.sub(
        r'(?<![a-z\d^}])[{]([\d=,\\._+\-*/()]+)[}](?![\d])', '\\1', ans_true_text)

    if ' answer is' in ans_all:
        end = '答案为：' + ans_all.split(' answer is')[-1] + '。'

    elif '<n># Answer<n>' in ans_all:
        end = '答案为：' + ans_all.split('<n># Answer<n>')[-1] + '。'

    else:
        end = ''

    split_n = ans_all.split('<n>')

    if len(split_n) > 10 and ans_all.count('<n>'.join(split_n[-3:-1])) > 8 and len('<n>'.join(split_n[-3:-1])) > 15:
        return False
    flag = re.findall(
        r'(?<![\d*+\-/{^(])(?:(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+))(?![}\d).+\-*/=√₀²³‰¼½¾_×¬^,!:±×÷∶∧∨∑∏∪∷√≡≠><≤≥])', ans_true_text)
    ans_flag = False
    if len(flag) == 1 and flag[0] == ans_true_text:

        q_text = replace_fraction(q_text)
        q_text = re.sub(
            r'(?<![a-z\d^}])[{(]+((?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+\_\d+)|(?:[-]?\d+))[})]+(?![\d])', '\\1', q_text)

        q_num_lst = re.findall(
            r'(?<![\d*+\-/({\[^><≤≥_√])(?:(?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+))(?![}\d+)\]\}\-*/=√₀²³‰¼½¾_×¬^,!:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', q_text)

        ans_true_text = replace_transmean(ans_true_text)

        if not end:
            if '<n>' in ans_all:
                end = split_n[-1]

                if end and not has_numbers(end) and not re.findall(r'答案为|answer', end):
                    end = split_n[-2]
                    if end and not has_numbers(end) and len(split_n) > 2 and not re.findall(r'答案为|answer', end):
                        end = split_n[-3]
            else:
                end = ans_all

        if '=' in end:
            end = '答案为' + end.split('=')[-1] + '。'
        end = end.replace('\\%', '%').replace('\%', '%').replace('\\leq', '≤').replace(
            '\\le', '≤').replace('\\geq', '≥').replace('\\ge', '≥')

        try:

            frac_pi = re.findall(r'\\frac\s*\{[^}]*?pi[^}]*?\}\s*\{[^}]*?\}', end)
            if frac_pi and '度数' in end:
                frac_pi_text = frac_pi[-1].replace(' ', '')
                numerator = re.findall(r'(?<=\\frac\{)[^}]*?(?=\})', frac_pi_text)[0].replace('\\pi', 'math.pi')
                denominator = re.findall(r'(?<=\}\{)[^}]*?(?=\})', frac_pi_text)[0]

                degrees = eval(numerator) / eval(denominator) * 180 / math.pi
                end = end.replace(frac_pi[-1], str(degrees))

            end = end.replace('\\frac ', '\\frac').replace('\\pi', 'π')
            end = replace_fraction(end)
            end = re.sub(
                r'(?<![a-z\d^])[{]+([-]?\d+\.?\d+/\d+\.?\d+)[}]+(?![\d])', '\\1', end)
            end = re.sub(
                r'(?<![a-z\d^}])[{]+((?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+\_\d+)|(?:[-]?\d+))[}]+(?![\d])', '\\1', end)
            end = re.sub(
                r'(?<![a-z\d^}])[(]+((?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+\_\d+)|(?:[-]?\d+))[)]+(?![\d])', '\\1', end)
            end = re.sub(r'\\sqrt\s*\{\[\}(?=\d+\])', r'\\sqrt[', end)
            end = re.sub(r'(\\sqrt)\s*\[\s*(\d+)s*\](\d+)', r'\1[\2]{\3}', end)
            end = re.sub(
                r'(?<=[.*+\-/(\(\)\{\}^\[\]_=√₀²³‰¼½¾×¬^!π:±÷∶∧∨∑∏∪∷√≡≠><≤≥])\s+(?=[a-zA-Z\d])', '', end)
            end = re.sub(
                r'(?<=\d)\s+(?=[.*+\-/(\(\)\{\}^\[\]_=√₀²³‰¼½¾×¬^!π:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', '', end)
            if '答案为' in end:
                match_regex_ans = '(?<![a-zA-Z.\d*+\-/({^\[_><≤≥√])' + ans_true_text + '(?![a-zA-Z}\d.+\-*/=)\]\}\\\√₀²³‰¼½¾_×¬^!π±÷∶∧∨∑∏∪∷√≡≠><≤≥])'
                ans_gen = re.findall(
                    r'(?<![a-zA-Z\d*+\-/({^\[_><≤≥√])(?:(?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+))(?![a-zA-Z}\d+)\-*/=_\]\}√₀²³‰¼½¾×¬^!π:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', end)

            else:
                match_regex_ans = '(?<![\d*+\-/({^\[_><≤≥√])' + ans_true_text + '(?![}\d.+\-*/=)\]\}\\\√₀²³‰¼½¾_×¬^!π±÷∶∧∨∑∏∪∷√≡≠><≤≥])'
                ans_gen = re.findall(
                    r'(?<![\d*+\-/({^\[_><≤≥√])(?:(?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+))(?![}\d+)\-*/=_\]\}√₀²³‰¼½¾×¬^!π:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', end)
            if not ans_gen:
                ans_gen = re.findall(
                    r'(?<![\d*+\-/({^\[_])(?:(?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+))(?![}\d+)\-*/=\]\}√₀²³‰¼½¾×¬^!π:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', end)

            # Convert a value containing % to a decimal
            frac_2 = re.findall(
                r'(?<![\d*+\-/({^\[_])(?:(?:[-]?\d+\.\d+)|(?:[-]?\d+))%', end)

            if frac_2:
                for value in frac_2:
                    ans_gen.insert(0, str(int(value.replace('%', '')) / 100))

            if 'sqrt' in end:
                sqrt = re.findall(r'(?<![\d{+\-*/])\\sqrt\{[^}]*?\}(?![}+\-*/])', end)
                for s_value in sqrt:
                    ans_gen.insert(0, s_value)

            ans_end = re.findall(match_regex_ans, end)
        except Exception as e:
            ans_end = ''
            print('Error: ', e)

        num_t = re.findall(r'(?:[-]?\d+\s*[、，,]){2,}[-]?\s*\d+', end)

        flag1 = True

        if bool(re.match(r'^-?\d+$', ans_true_yuan)) and num_t and (len(num_t[-1].split('、')) > int(ans_true_yuan) or len(num_t[-1].replace('，', '').split(',')) > int(ans_true_yuan)):
            flag1 = False
        if ans_true_yuan.isdigit() and len(ans_gen) == 2 and ans_gen[-1] not in q_num_lst and ans_gen[-1] != ans_true_yuan:
            flag1 = False

        if ans_true_yuan.isdigit() and len(ans_end) > 5:
            flag1 = False
        if flag1:
            if ans_end:
                ans_flag = True
            else:
                try:
                    if ans_gen:
                        ans_true_text = ans_true_text.replace(',', '').replace(' ', '').replace('\\', '')
                        num_v = max(-1, len(ans_gen)-4)
                        if len(ans_true_text.split('/')) == 2:
                            num = ans_true_text.split('/')[0]
                            den = ans_true_text.split('/')[1]
                            ans_true_text = float(num) / float(den)
                        if '.' in str(ans_true_text):
                            len_num = max(len(str(ans_true_text).split('.')[1]), 5)
                        else:
                            len_num = 5
                        if type(ans_true_text) == str:
                            ans_true_text = eval(ans_true_text)

                        for vi in range(len(ans_gen)-1, num_v, -1):
                            value = ans_gen[vi]

                            if value in q_num_lst and value != ans_true_yuan:
                                continue
                            value = value.replace(',', '').replace(' ', '')
                            if 'sqrt' in value:
                                value = value.replace('\\', '')
                                value = re.sub(r'sqrt\{([^}]*)\}', 'math.sqrt(\\1)', value)

                            if '/' in value:
                                num = value.split('/')[0]
                                den = value.split('/')[1]
                                value = float(num) / float(den)

                            if type(value) == str:
                                value = eval(value)

                            if round(value, len_num) == round(ans_true_text, len_num):
                                ans_flag = True
                except Exception as e:
                    print('eval Error:', e)

    elif re.findall(r'^((?:(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-]?\d+))[,]?)+$', ans_true_yuan):

        ans_true_text = ans_true_yuan.replace(',', '')
        if not end:
            end = split_n[-1]
        end_text = end.replace(' and ', ',').replace(' or ', ',').replace(',', '').replace(' ', '')

        match_regex_ans = '(?<=[^\d+\-*/^><≤≥])' + ans_true_text + '(?![\d.+\-*/=√₀²³‰¼½¾_×¬^!±×÷∧∨∑∏∪∷√≡≠><≤≥])'
        ans_end = re.findall(match_regex_ans, end_text)
        if ans_end:
            ans_flag = True
        else:
            ans_flag = False
    elif '\\text' in ans_true_yuan:
        if not end:
            end = split_n[-1]
        ans_true_text = re.findall(r'(?<=\\text\{)[^()}]+(?=\})', ans_true_yuan)
        if ans_true_text:
            end_lst = re.findall(ans_true_text[0], end, re.IGNORECASE)
            if not end_lst or len(end_lst) > 5 or '无法确定' in end:
                ans_flag = False
            else:
                ans_flag = True

        if not re.findall(r'[A-F]+', end) and len(split_n) > 1:
            end = split_n[-2]

        ans_true_text = re.findall(r'(?<=^\\text\{\()[A-F]+(?=\)\}$)', ans_true_yuan)
        if ans_true_text:
            flag = re.findall(f'[（(]{ans_true_text[0]}[)）]', end)
            if not flag:
                flag = re.findall(f'[^a-zA-Z]{ans_true_text[0]}[^a-zA-Z+\-*/]', end)
            if flag:
                ans_flag = True
            else:
                ans_flag = False

    else:
        if not end:
            end = split_n[-1]
        if not re.findall(r'[\d]+', end) and len(split_n) > 1:
            end = split_n[-2]

        if '=' not in ans_true_text and '=' in end:
            end = '答案为' + end.split('=')[-1] + '。'
        end = end.replace('π', '\\pi').replace(' ', '')
        end = re.sub(r'√(?=\d)', r'\\sqrt', end)

        try:
            end = _fix_sqrt(end)
        except Exception as e:
            print('Error: ', e)

        end = replace_fraction(end)

        end = re.sub(
            r'(?<![a-z\d^])[{]+([-]?\d+\.?\d+/\d+\.?\d+)[}]+(?![\d])', '\\1', end)
        end = re.sub(
            r'(?<![a-z\d^}]){((?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:\d+\_\d+)|(?:[-]?\d+))}(?![\d])', '\\1', end)
        end = re.sub(
            r'(?<![a-z\d^}]){((?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:\d+\_\d+)|(?:[-]?\d+))}(?![\d])', '\\1', end)
        end = re.sub(
            r'(?<![a-z\d^}])[(]+((?:[-]?\d+\.?\d+/\d+\.?\d+)|(?:[-]?\d+/\d+)|(?:[-]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:\d+\_\d+)|(?:[-]?\d+))[)]+(?![\d])', '\\1', end)

        end = re.sub(
            r'(?<![a-z\d^}])[{(](\([\d=,._+\-*/()]+\))[})](?![\d])', '\\1', end)
        end = re.sub(
            r'(?<![a-z\d^}])[{]([\d=,\\._+\-*/()]+)[}](?![\d])', '\\1', end)

        if 'x\in' in ans_true_yuan:
            ans_true_yuan = ans_true_yuan.replace('x\in', '')
        complex_flag = re.findall(r'^[-]?\d+[+\-]\d+[ij]$', ans_true_yuan)
        if 'frac' in ans_true_yuan:
            ans_true_lst = [ans_true_yuan, ans_true_text]
        elif ans_true_text == '\\cotx':
            ans_true_lst = [ans_true_text, '1/tan(x)', '1/tanx', '\\cot x', '1/tan x']
        elif complex_flag == [ans_true_yuan]:
            z = complex(ans_true_yuan.replace('i', 'j'))
            virtual_first_str = f"{int(z.imag)}j+{int(z.real)}"
            virtual_first_str = virtual_first_str.replace('+-', '-')
            if 'i' in ans_true_yuan:
                virtual_first_str = virtual_first_str.replace('j', 'i')
            ans_true_lst = [ans_true_yuan, virtual_first_str]

        elif re.findall(r'^[a-zA-Z]=\d+$', ans_true_yuan):
            ans_true_lst = [ans_true_yuan,
                            re.findall(r'\d+', ans_true_yuan)[0]]
        elif re.findall(r'^\d+_\d+$', ans_true_yuan):
            ans_true_lst = [ans_true_yuan, ans_true_yuan.split('_')[0]]
        else:
            ans_true_lst = [ans_true_yuan]
        for value in ans_true_lst:
            value = value.replace(' ', '')
            value = replace_transmean(value)
            match_regex_ans = '(?<![\d+\-*/^><≤≥∩∪^{])' + value + '(?![\d+\-*/=√₀}²³‰¼½¾_×¬^!±×÷∧∨∑∏∪∷√≡≠><≤≥])'
            ans_end = re.findall(match_regex_ans, end)

            if ans_end:
                ans_flag = True

    return ans_flag


def eval_gen_res(txt_files_path, origin_file_path, txt_eval_res_dir=None):

    if not txt_files_path or not os.path.exists(txt_files_path):
        logging.error("The input evaluation path is incorrect.")
        return

    if not origin_file_path or not os.path.exists(origin_file_path):
        logging.error("The original file path is incorrect.")
        return

    try:

        content_qa = load_a_file(origin_file_path)
        content_yuan_all_dict = {}
        for i in range(len(content_qa)):
            g = content_qa[i]
            if g == "":
                continue
            g = re.sub(r'(<n>)+', '<n>', g).strip()
            g = g.split('[SEP]')

            content_yuan_all_dict[g[0].strip()] = g[1].replace(' ', '')

        len_ori = len(content_qa)

        if os.path.isfile(txt_files_path):
            qa_content = load_a_file(txt_files_path)
        else:
            qa_content = process_gen_files(txt_files_path, len_ori)
    except Exception as e:
        print('Error: ', e)

    text_answer_true, text_answer_false, text_answer_no = [], [], []
    for i in range(len(qa_content)):
        qa_content[i] = qa_content[i].strip()
        if not qa_content[i]:
            continue

        q_text = qa_content[i].split('[SEP]')[0]
        q_text = re.sub(r'(<n>)+', '<n>', q_text).strip()
        ans_true = ''
        for key in list(content_yuan_all_dict.keys()):

            if q_text in key or key in q_text:
                ans_true = content_yuan_all_dict[key]
                break

        if not ans_true:
            print(f'No correct answer found for question: "{q_text}".')
            continue
        try:
            ans_flag = process_content(qa_content[i], ans_true)
        except Exception as e:
            ans_flag = False
        if ans_flag:
            text_answer_true.append(qa_content[i] + '[EOD]' + ans_true + '\n')
        else:
            text_answer_false.append(qa_content[i] + '[EOD]' + ans_true + '\n')

    accuracy = len(text_answer_true) / (len(text_answer_true) + len(text_answer_false))

    print(f'Number of correct answers:{len(text_answer_true)}')
    print(f'Number of incorrect answers:{len(text_answer_false)}')
    print('accuracy:', accuracy)
    try:
        if txt_eval_res_dir:
            if not os.path.exists(txt_eval_res_dir):
                os.makedirs(txt_eval_res_dir)
            save_a_file(text_answer_true, os.path.join(txt_eval_res_dir, "Math_res_true.txt"))
            save_a_file(text_answer_false, os.path.join(txt_eval_res_dir, "Math_res_false.txt"))
            save_a_file(text_answer_no, os.path.join(txt_eval_res_dir, "Math_res_no.txt"))
    except Exception as e:
        print('save Error: ', e)


def main():
    global file_type_list
    file_type_list = ['jsonl']

    origin_file_path = "<Specify path>"
    eval_file_path = "<Specify path>"
    txt_eval_res_dir = "<Specify path>"

    eval_gen_res(eval_file_path, origin_file_path, txt_eval_res_dir)


if __name__ == "__main__":

    main()
