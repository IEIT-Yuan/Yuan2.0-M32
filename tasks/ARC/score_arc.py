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


def get_file_list(folder, file_type_list):
    filelist = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if file_type in file_type_list:
                file_fullname = os.path.join(dirpath, file)
                filelist.append(file_fullname)

    return filelist


def process_gen_files(gen_path_dir, len_ori):
    txt_files_lst = get_file_list(gen_path_dir, file_type_list)
    try:
        txt_files_lst = sorted(txt_files_lst, key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    except Exception as e:
        print(e)

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


def get_ans_gen_letter(qa_text):
    if not qa_text:
        return qa_text
    a_text = qa_text.strip().split('[SEP]')[1]
    q_text = qa_text.split('[SEP]')[0].replace('．', '.')

    flag_q = re.findall(
        r'(?:错误)|(?:incorrect)|(?:not true)|(?:False)|(?:无法推断)', q_text)
    if flag_q:
        flag_a = False
    else:
        flag_a = True
    a_text = a_text.replace('$', '').replace('Ｃ', 'C').replace(
        'Ａ', 'A').replace('Ｂ', 'B').replace('Ｄ', 'D')
    a_text = re.sub(r'\\rm\s*\{([A-Z]+)\}', '\\1', a_text)
    a_text = re.sub(r'\\mathrm\s*\{([A-Z]+)\}', '\\1', a_text)
    a_text = re.sub(r'\\boxed\s*\{\s*([A-Z]+)\s*\}\s*', '\\1', a_text)
    a_text = re.sub(r'\\text\s*\{[\(]?([A-Z]+)[\)]?\}', '\\1', a_text)
    a_text = re.sub(r'<font[^>]*?>([A-K]+)</font>', '\\1', a_text)
    a_text = re.sub(r'\\textbf\s*\{\(([A-Z]+)\)\}', '\\1.', a_text)
    a_text = re.sub(r'\\mathbf\s*\{([^{]*?)\}', '\\1', a_text)
    ans_text = re.findall(
        r'(?:correct|most appropriate|most accurate|plausible|reasonable|best|closest) (?:answer|option|Option|statement) (?:is|would be)[option\s<n>:*\(]*?[A-K]\s*[\),:\.]', a_text)

    if not ans_text:
        ans_text = re.findall(
            r'(?:correct|most appropriate|most accurate|plausible|reasonable|best|closest) answer [\sa-z]*? "[^"]*?"[isoption\s<n>:*\(]*?[A-K]\s*[\),:\.]', a_text)
    else:
        ans_text = re.sub(
            r'(?:correct|most appropriate|most accurate|plausible|reasonable|best|closest) (?:answer|option|Option|statement) (?:is|would be)[option\s<n>:*\(]*?([A-K])\s*[\),:\.]', '\\1', ans_text[-1])

    if not ans_text:
        ans_text = re.findall(
            r'(?:option|Option) [A-K]\s*[\),:\.][\sa-z]*? "[^"]*?"[a-z\s]*?the correct answer', a_text)
    else:
        ans_text = re.sub(
            r'(?:correct|most appropriate|most accurate|plausible|reasonable|best) answer [\sa-z]*? "[^"]*?"[isoption\s<n>:*\(]*?([A-K])\s*[\),:\.]', '\\1', ans_text[-1])
    if not ans_text:
        ans_text = re.findall(
            r'(?:option|Option) [A-K](?:\s*[\.\)][\s:a-zA-Z]{0,10}|) is the correct answer', a_text)
    else:
        ans_text = re.sub(
            r'(?:option|Option) ([A-K])\s*[\),:\.][\sa-z]*? "[^"]*?"[a-z\s]*?the correct answer', '\\1', ans_text[-1])
    if not ans_text:
        ans_text = re.findall(
            r' answer (?:is|would be) .*?[A-K]\s*[\)\.]', a_text)
    else:
        ans_text = re.sub(
            r'(?:option|Option) ([A-K])(?:\s*[\.\)][\s:a-zA-Z]{0,10}|) is the correct answer', '\\1', ans_text[-1])
    if not ans_text:
        if not flag_a:
            ans_text = re.findall(
                r'incorrect statement (?:is|would be)[option\s<n>:*\(]*?[A-K]\s*[\),:\.]', a_text)

            if ans_text:
                ans_true = []
                for al in range(len(ans_text)):
                    choice_sub = re.sub(
                        r'incorrect statement (?:is|would be)[option\s<n>:*\(]*?([A-K])\s*[\),:\.]', '\\1', ans_text[al])
                    ans_true.append(choice_sub)
                if ans_true:
                    ans_true = ''.join(sorted(list(set(ans_true))))
                    ans_text = [ans_true]
                    print(ans_true)
    else:
        ans_text = [re.sub(r' answer (?:is|would be) [^\.]*?([A-K])\s*[\)\.]', '\\1', ans_text[-1])]
    if not ans_text:
        split_n = a_text.split('<n>')
        ans_text = re.findall(r'(?:option|answer|Option|answer|)[a-z\s]*?[A-K][\s\.,:\)]', split_n[-1])
        if not ans_text and len(split_n) >= 2:
            ans_text = re.findall(r'(?:option|answer|Option|answer|)[a-z\s]*?[A-K][\s\.,:\)]', split_n[-2])
        if ans_text:
            ans_text = [re.sub(r'(?:option|answer|Option|answer|)[a-z\s]*?([A-K])[\s\.,:\)]', '\\1', ans_text[-1])]
    if re.findall(r'none of the [a-z\s*] options are correct.', a_text):
        ans_text = None
    if ans_text:
        try:
            if re.findall(r'[A-K]+', ans_text[-1]) != ans_text:
                ans = re.findall(r'[A-K]', ans_text[-1])[-1]
            else:
                ans = ans_text
            ans = ''.join(sorted(list(set(ans))))
            return ans
        except Exception as e:
            print(e)
            print(ans_text)
            return None
    else:
        return None


def eval_gen_res(txt_files_path, origin_file_path=None, txt_eval_res_dir=None):

    if not txt_files_path or not os.path.exists(txt_files_path):
        logging.error("The input evaluation path is incorrect.")
        return

    if not origin_file_path or not os.path.exists(origin_file_path):
        logging.error("The original file path is incorrect.")
        return

    try:
        file_name = '_'.join(os.path.basename(origin_file_path).split('.')[0].split('_')[:-1])
        content_qa = load_a_file(origin_file_path)
        content_yuan_all_dict = {}
        for i in range(len(content_qa)):
            g = re.sub(r'(<n>)+', '<n>', content_qa[i]).strip()
            if g == "":
                continue
            g = g.split('[SEP]')
            content_yuan_all_dict[g[0].strip()] = g[1].replace(' ', '')
        len_ori = len(content_qa)

        print(f'len_ori:{len_ori}')

        if os.path.isfile(txt_files_path):
            qa_content = load_a_file(txt_files_path)
        else:
            qa_content = process_gen_files(txt_files_path, len_ori)
    except Exception as e:
        print('Error: ', e)

    text_answer_true, text_answer_false = [], []

    content_false_q = []
    for i in range(len(qa_content)):
        qa_content[i] = qa_content[i].strip()
        if not qa_content[i]:
            continue
        qa_content[i] = re.sub('(\s*<n>\s*)+?$', '', qa_content[i])
        q_text = qa_content[i].split('[SEP]')[0].strip()
        ans_true = ''
        for key in list(content_yuan_all_dict.keys()):
            if q_text in key or key in q_text:
                ans_true = content_yuan_all_dict[key]
                qa_text = key + '[SEP]' + ans_true + '\n'
                break
        if not ans_true:
            print(f'No correct answer found for question: "{q_text}".')
            continue

        ans_true = ans_true.replace('Answer:', '').replace('Answer：', '').replace('.', '').replace('Answers:', '').replace('Answers：', '').strip()

        qa_content[i] = qa_content[i].split('[EOD]')[0]
        try:
            ans_gen = get_ans_gen_letter(qa_content[i])
        except Exception as e:
            ans_gen = None

        if ans_true == ans_gen:
            text_answer_true.append(qa_content[i] + ' [EOD] ' + ans_gen + ' [ANS] ' + ans_true + '\n')
        else:
            if ans_gen:
                text_answer_false.append(qa_content[i] + ' [EOD] ' + ans_gen + ' [ANS] ' + ans_true + '\n')
            else:
                text_answer_false.append(qa_content[i] + ' [EOD] ' + ' None ' + ' [ANS] ' + ans_true + '\n')
        content_false_q.append(qa_text)
    accuracy = len(text_answer_true) / (len(text_answer_true) + len(text_answer_false))
    print(f'Number of correct answers:{len(text_answer_true)}')
    print(f'Number of incorrect answers:{len(text_answer_false)}')
    print('accuracy:', accuracy)
    try:
        if txt_eval_res_dir:
            if not os.path.exists(txt_eval_res_dir):
                os.makedirs(txt_eval_res_dir)
            save_a_file(text_answer_true, os.path.join(txt_eval_res_dir, f"{file_name}_true.txt"))
            save_a_file(text_answer_false, os.path.join(txt_eval_res_dir, f"{file_name}_false.txt"))
        with open(os.path.join(txt_eval_res_dir, f"{file_name}_qa_false.txt"), 'w', encoding='utf-8') as f:
            f.writelines(content_false_q)
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
