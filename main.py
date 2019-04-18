import csv
import os
import re
from collections import Counter

import nltk
import pandas as pd
from scipy import spatial

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

ATTACHMENT_PATH = "Attachments"


def read_csv_file(filepath, usecols):
    dataset = pd.read_csv(filepath, usecols=usecols)
    return dataset


def get_matched_summaries():
    summaries_path = os.path.join(ATTACHMENT_PATH, "Summaries.csv")

    summaries = read_csv_file(summaries_path,
                              usecols=["adsh", "full_summary", "full_summary_len", "expense_summary", "expense_len"])
    total_summaries = summaries.adsh.tolist()

    non_summaries_path = os.path.join(ATTACHMENT_PATH, "Non-Summaries")
    non_summaries_file_list = os.listdir(non_summaries_path)
    non_summaries_file_list = [os.path.splitext(os.path.basename(file_path))[0] for file_path in
                               non_summaries_file_list]

    mapping_path = os.path.join(ATTACHMENT_PATH, "Mapping.csv")
    mapping = read_csv_file(mapping_path, usecols=["accession_xbrl", "accession_not_xbrl"])

    new_mapping = mapping[(mapping['accession_not_xbrl'].isin(non_summaries_file_list)) & (
        mapping['accession_xbrl'].isin(total_summaries))]

    return new_mapping, mapping, summaries


def get_lines_from_paragraph(paragraph):
    words_output = list()
    lines = list()
    splitted_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
    for line in splitted_sentences:
        words_output.append(remove_articles_preposition(line.strip()))
        lines.append(line.strip())
    return words_output, lines


def remove_articles_preposition(sentence):
    tokens = nltk.word_tokenize(sentence)
    # remove special characters
    tokens = list(filter(lambda x: x, map(lambda x: re.sub(r'[^A-Za-z0-9]+', '', x), tokens)))

    # remove articles
    tokens = [token for token in tokens if token.lower() not in ['a', 'an', 'the']]
    tagged = nltk.pos_tag(tokens)

    # remove prepositions and stem the words
    stemmer = nltk.stem.SnowballStemmer('english')

    clean_word_list = [stemmer.stem(x) for (x, y) in tagged if y != 'IN']
    return clean_word_list


def read_file(filepath):
    with open(filepath, "r", encoding="utf8", errors='ignore') as input_file:
        content = input_file.read()

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(content.strip())

    # remove sentences not ending with full stop
    sentences = [re.sub('\s+', ' ', sentence) for sentence in sentences if sentence[-1] is '.' or sentence[-1] is '?']
    return sentences


def calculate_word_vector_count(summary_word_vector, non_summary_wordvector):
    total_score = list()
    for line in summary_word_vector:
        cosine = list()
        for line2 in non_summary_wordvector:
            combine_word_vector = list(set(line + line2))
            summary_count_vector = get_word_count(line, combine_word_vector)
            non_summary_count_vector = get_word_count(line2, combine_word_vector)

            cosine.append(get_cosine_similarity(summary_count_vector, non_summary_count_vector))
        total_score.append(max(cosine))
    total = get_average_similarity(total_score)
    return total, total_score


def get_average_similarity(total):
    return sum(total) / len(total)


def get_cosine_similarity(summary_count_vector, non_summary_count_vector):
    # 1-(1-cos(Theta)) dont get confused here
    cosine_value = 1 - spatial.distance.cosine(summary_count_vector, non_summary_count_vector)
    if cosine_value >= 0.90:
        result = 1
    else:
        result = 0
    return result


def get_word_count(to_count_list, word_list):
    counts = Counter()
    counts.update(to_count_list)
    for word in word_list:
        if not counts.get('word', None):
            counts.update({word: 0})
    return get_ordered_vector(counts, word_list)


def get_ordered_vector(counter, vector):
    return [counter.get(key, 0) for key in vector]


def add_rows_to_output_csv(output_rows):
    summaries_path = os.path.join(ATTACHMENT_PATH, "Summaries.csv")
    with open(summaries_path, 'r') as csvinput:
        with open('Output/summaries_output.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all_rows = []
            row = next(reader)
            row.append('number_of_sentences_in_summary')
            row.append('number_of_sentences_in_non_summary')
            row.append('similarity')
            all_rows.append(row)

            for row in reader:
                for output_row in output_rows:
                    if row[0] == output_row.get("summary_indentifier"):
                        row.append(output_row.get("number_of_sentences_in_summary"))
                        row.append(output_row.get("number_of_sentences_in_non_summary"))
                        row.append(output_row.get("similarity"))
                        all_rows.append(row)

            writer.writerows(all_rows)


def make_raw_files(filename, non_summary_identity, line_wise_similarity, summary_lines):
    header = ["adsh", "non_summary_file" "sentence", "similarity"]
    with open('Raw/' + filename + ".csv", 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        all_rows = []
        for index, line in enumerate(summary_lines):
            all_rows.append([filename, non_summary_identity, line, line_wise_similarity[index]])
        wr.writerows(all_rows)


def main():
    output_rows = list()
    new_mapping, mapping, summaries = get_matched_summaries()
    for index, row in new_mapping.iterrows():
        summary_file_identity, non_summary_file_identity = row['accession_xbrl'], row['accession_not_xbrl']
        full_summary = summaries.loc[summaries["adsh"] == summary_file_identity, 'full_summary'].iloc[0]
        summary_word_vector, summary_lines = get_lines_from_paragraph(full_summary)

        number_of_summary_lines = len(summary_word_vector)
        non_summary_file_path = os.path.join(ATTACHMENT_PATH, "Non-Summaries", non_summary_file_identity + '.txt')

        non_summary_lines = read_file(non_summary_file_path)
        number_of_non_summary_lines = len(non_summary_lines)

        non_summary_word_vector = list()
        for non_summary_line in non_summary_lines:
            non_summary_word_vector.append(remove_articles_preposition(non_summary_line))

        total_similarity, line_wise_similarity = calculate_word_vector_count(summary_word_vector,
                                                                             non_summary_word_vector)
        make_raw_files(summary_file_identity, non_summary_file_identity, line_wise_similarity, summary_lines)

        output_rows.append(dict(summary_indentifier=summary_file_identity,
                                similarity=total_similarity,
                                number_of_sentences_in_summary=number_of_summary_lines,
                                number_of_sentences_in_non_summary=number_of_non_summary_lines))
    print(output_rows)
    add_rows_to_output_csv(output_rows)


if __name__ == "__main__":
    main()
