import copy
import csv


def read_conllu(filename):
    """Read a conllu file return/yield it as a list of lists of edge lists.

    Here is an example CoNLL-U fragment:
    -------------------------------------------------------------------
    1	You	you	PRON	_	_	3	nsubj	_	_
    2	are	be	AUX	_	Number=Plur	3	cop	_	_
    3	late	late	ADJ	_	_	0	root	_	SpaceAfter=No
    -------------------------------------------------------------------
    which should be represented as:
    [(0, None, None, ('ROOT', '_', '_', '_')),
     (3, 1, 'nsubj', ('You', 'you', 'PRON', '_')),
     (3, 2, 'cop', ('are', 'be', 'AUX', 'Number=Plur'))
     (0, 3, 'root', ('late', 'late', 'ADJ', '_'))
    ]
    """

    # Outer list to be returned
    result = []

    root = ('0', None, None, ('ROOT', '_', '_', '_'))
    try:
        # Iterate over the ConLL-U file
        with open(filename, mode='r', encoding='utf-8') as f:
            sentence = [root]
            for line in f:
                line = line.strip()

                # Skip comment lines
                if line.startswith('#'):
                    continue

                # Empty line indicates new sentence. Add it to the list, and reset the sentence to contain only root
                if line == '':
                    # Skip empty line after comments
                    if sentence == [root]:
                        continue
                    result.append(sentence)
                    sentence = [root]
                    continue

                # Otherwise, we want to use the line, parse with tab delimiter
                line = line.split('\t')

                # ignore if the line is not correct format (not exactly 10 fields for non-comment lines)
                if not 10 >= len(line) or not 8 <= len(line):
                    print("Error: Some non-comment lines have an unexpected number of fields (expected 10, separated by tab)")
                    continue

                # Skip multi-word tokens and empty tokens
                if '-' in line[0] or '.' in line[0]:
                    continue

                # Build the word based on current line, append to sentence
                word = (line[6], line[0], line[7], (line[1], line[2], line[3], line[5]))
                sentence.append(word)

            # Add the last sentence
            if sentence != [root]:
                result.append(sentence)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    else:
        return result


def write_conllu(edge_lists, filename):
    """Take a list of lists of edge lists and write the equivalent CoNLL-U.

    Given such a list, compatible with the output of
    read_conllu() above, output it as a CoNLL-U file.
    Reading a treebank with read_conllu() and writing with
    write_conllu() should produce the same CoNLL-U information,
    except you can drop enhanced dependencies, information
    in the MISC field, and null and multi-word tokens.
    """
    path = f"output_files/{filename}"

    with open(path, mode='w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for sentence in edge_lists:
            for edge in sentence[1:]:
                # If an edge has an invalid format, skip it
                if len(edge) != 4 or len(edge[3]) != 4:
                    print("Error: Invalid edge format, expected: (idx_head, idx_dep, deprel, (form, lemma, pos, morph))")
                    continue
                # Else, write it into the file
                tsv_writer.writerow([edge[1], edge[3][0], edge[3][1], edge[3][2], '_', edge[3][3], edge[0], edge[2], '_', '_'])


def is_non_proj(sent):
    """
    Return the head and dependent of a crossing relation if 'sent' is non-projective.
    """

    """
    projectivity: An arc from a head to a dependent is said to be projective if there is a path 
                    from the head to every word that lies between the head and the dependent in the sentence. 
    non-projectivity: for each head and dependent, exists one x between head and dependent such that  
                        x doesn't have head as ancestor
    """

    # Make sure the first edge represents the root
    if sent[0] != ('0', None, None, ('ROOT', '_', '_', '_')):
        print("Error: The first edge should be the root")
        return None, None

    # Create dictionary of pairs in form: {dependent : head}
    edge_list = dict()

    # Make sure all edges match the expected format
    for word in sent[1:]:
        if len(word) != 4 or len(word[3]) != 4:
            print("Error: The sentence contains an invalid edge. Expected format: (idx_head, idx_dep, deprel, (form, lemma, pos, morph))")
            return None, None
        else:
            edge_list[int(word[1])] = int(word[0])

    # Iterate over every dependent-head pair
    for dep in edge_list:
        original_head = edge_list.get(dep)
        ancestors = []

        # Skip root
        if dep is None:
            continue

        # For every word between dependent and head, trace the path to root
        for word_in_between in range(min(int(dep), int(original_head)) + 1, max(int(dep), int(original_head))):
            head = edge_list[word_in_between]
            while head != 0:
                ancestors.append(head)
                head = edge_list[head]
            ancestors.append(0)

            # If the path to root did not include the original head, the structure is non-projective
            if original_head not in ancestors:
                return original_head, dep

            # Clear list for the next word
            ancestors.clear()

    return None, None


def projectivize(edge_list):
    """Convert given dependency tree to a pseudo-projective dependency tree.

    Both input and output values should be the edge list structure
    defined in p1.py. If the given edge list is projective, it should
    be returned as is.
    """

    result = []

    # To projectivize, we want to access the crossing edge to be deleted, and the head of the original head

    # Iterate over the list of sentences
    for sent in edge_list:
        # Check if sentence is projective (using the method from p3, return values would be not None)
        head, dep = is_non_proj(sent)

        # If non-projective, iterate to find indices of the crossing edge, and head of the original head
        # Repeat until no crossing edges remain
        while head and dep:
            head_of_head = sent[head]
            # Update the head and dependency relation of the crossing edge accordingly
            # Convert to list to bypass tuple immutability
            tmp = list(sent[dep])
            tmp[0] = head_of_head[0]
            tmp[2] = head_of_head[2] + '.' + tmp[2]
            sent[dep] = tuple(tmp)

            head, dep = is_non_proj(sent)

        # If projective, keep as-is
        result.append(sent)

    return result


def find_dep_by_head_and_relation(sentence, head, dep_relation):
    """
    Helper function,
    Returns the dependent (index) of the edge with a given head and dependency relation
    If it does not exist, returns -1 (fallback value, it will never be returned in our use case)
    """
    for node in sentence:
        if node[0] == head and node[2] == dep_relation:
            return node[1]
    return -1


def deprojectivize(edge_list):
    """Given a pesudo-projective dependency tree, return a non-projective tree.

    Both input and output values should be the edge list structure
    defined in p1.py. If the given edge list has any complex
    dependency labels generated by projectivize() in p4.py,
    convert it back to a non-projective dependency tree. Otherwise,
    the input should be returned as is.
    """

    # Clone the original edge list, to allow permutation without affecting the original
    clone = copy.deepcopy(edge_list)

    # Iterate over the sentences
    for sent in clone:

        # Make sure the first node represents the root
        if sent[0] != ('0', None, None, ('ROOT', '_', '_', '_')):
            print("Error: The first edge should be the root, skipping...")
            continue

        # For each sentence, iterate over the edges
        for i in range(len(sent)):
            edge = sent[i]

            # Make sure each edge matches the expected format
            if len(edge) != 4 or len(edge[3]) != 4:
                print('Error: the sentence has an invalid format, skipping...')
                break

            # Found a pseudo-projective label
            if edge[2] and '.' in edge[2]:
                # Update the edge: head becomes the dependent of the current head, whose label is the 'left-side' label
                # The label of the new edge will be the 'right-side' label
                left_rel, right_rel = edge[2].split('.')
                dep = find_dep_by_head_and_relation(sent, edge[0], left_rel)

                # Convert tuple to list to bypass immutability
                tmp = list(edge)
                tmp[0] = dep
                tmp[2] = right_rel

                # Convert back to tuple, and update the sentence
                sent[i] = tuple(tmp)

    return clone