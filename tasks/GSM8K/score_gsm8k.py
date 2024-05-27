import os
import re
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
    pattern = r'\\[d]?frac\{([\d,.]+)\}\{([\d,.]+)\}'
    replacement = r'\1/\2'
    replaced_text = re.sub(pattern, replacement, text)

    return replaced_text


def has_numbers(input_string):
    pattern = r'\d+'
    match = re.search(pattern, input_string)
    return bool(match)


def judge_ans_res(qa_text, ans_true):
    qa_text = re.sub('(\s*<n>\s*)+?$', '', qa_text)
    qa_text = re.sub(r'\\boxed{\s*([\d.,/]+)\s*}[.]?', '\\1', qa_text)
    ans_all = qa_text.split('[SEP]')[1].replace('<br>', '<n>').replace('\!', '').replace('\;', '').replace('\,', '')
    ans_all = ans_all.replace('$$', ' ').replace('$', '').replace(' ', '').replace('≈', '=')
    ans_true_yuan = ans_true
    ans_true = replace_transmean(ans_true)
    match_regex_ans = '(?<=[^0-9+\-*/])' + ans_true + \
        '(?![xyzXYZ\d.+\-*/=()√₀\²³‰¼½¾_×¬^,!:±×÷∶∧∨∑∏∪∷√≡≠><≤≥])'
    split_n = ans_all.split('<n>')
    end_2, ans_end_2 = '', []
    if '<n>' in ans_all:
        end = split_n[-1]
        if '=' not in end and 'answer is ' not in end:
            end_2 = '<n>' + split_n[-2].split('=')[-1]
            if end and not has_numbers(end):
                end = split_n[-2]
            if end and not has_numbers(end) and len(split_n) > 2:
                end = split_n[-3]
    else:
        end = ans_all
    try:
        if 'The answer is' in end:
            end = '答案为：' + end.split('The answer is')[-1]
        if '=' in end:
            end = '答案为：' + end.split('=')[-1]+'。'
        end = end.replace('\\%', '%').replace('\%', '%')
        end = replace_fraction(end)
        ans_gen = re.findall(
            r'(?:(?:[-+]?\d+\.?\d+/\d+\.?\d+)|(?:[-+]?\d+/\d+)|(?:[-+]?\d+\.\d+)|(?:\d{1,5}(?:,\d{3})+(?:\.\d+)*?)|(?:[-+]?\d+))(?![\d+\-*/=()√₀\²³‰¼½¾_×¬^:±÷∶∧∨∑∏∪∷√≡≠><≤≥])', end)
        if end_2:
            end_2 = replace_fraction(end_2)
            ans_end_2 = re.findall(match_regex_ans, end_2)

        ans_end = re.findall(match_regex_ans, end)
    except Exception as e:
        print('Error: ', e)

    ans_flag = False
    if ans_end:
        ans_flag = True
    else:
        if ans_end_2 and not ans_gen:
            ans_flag = True
        try:
            if ans_gen:
                ans_true = ans_true_yuan.replace(',', '').replace(' ', '')
                num_v = max(-1, len(ans_gen)-4)
                for vi in range(len(ans_gen)-1, num_v, -1):
                    value = ans_gen[vi]
                    value = value.replace(',', '').replace(' ', '')
                    if '/' in value:
                        num = value.split('/')[0]
                        den = value.split('/')[1]
                        value = float(num) / float(den)
                    if float(value) == float(ans_true):
                        ans_flag = float(value) == float(ans_true)
                        break

        except Exception as e:
            print('eval Error:', e)
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
            g = g.strip('\n')
            g = g.split('[SEP]')
            content_yuan_all_dict[g[0].strip()] = g[1].replace(' ', '')

        len_ori = len(content_qa)

        if os.path.isfile(txt_files_path):
            qa_content = load_a_file(txt_files_path)
        else:
            qa_content = process_gen_files(txt_files_path, len_ori)
    except Exception as e:
        print('Error: ', e)

    text_answer_true, text_answer_false = [], []

    for i in range(len(qa_content)):
        qa_content[i] = qa_content[i].strip()
        if not qa_content[i]:
            continue

        q_text = qa_content[i].split('[SEP]')[0].strip()

        ans_true = ''
        for key in list(content_yuan_all_dict.keys()):
            if q_text in key or key in q_text:
                ans_true = content_yuan_all_dict[key]
                break
        if not ans_true:
            print(f'No correct answer found for question: "{q_text}".')
            continue
        try:
            ans_flag = judge_ans_res(qa_content[i], ans_true)
        except Exception as e:
            ans_flag = False
        qa_content[i] = qa_content[i] + '[EOD]' + ans_true
        if ans_flag:
            text_answer_true.append(qa_content[i] + '\n')
        else:
            text_answer_false.append(qa_content[i] + '\n')

    accuracy = len(text_answer_true) / (len(text_answer_true) + len(text_answer_false))

    print(f'Number of correct answers:{len(text_answer_true)}')
    print(f'Number of incorrect answers:{len(text_answer_false)}')
    print('accuracy:', accuracy)
    try:
        if txt_eval_res_dir:
            if not os.path.exists(txt_eval_res_dir):
                os.makedirs(txt_eval_res_dir)
            save_a_file(text_answer_true, os.path.join(txt_eval_res_dir, "gsm_res_true.txt"))
            save_a_file(text_answer_false, os.path.join(txt_eval_res_dir, "gsm_res_false.txt"))
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
