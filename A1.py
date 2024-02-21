from functions import QueryEngine

DATA_FOLDERPATH = "data"

engine = QueryEngine(DATA_FOLDERPATH, debug=False)

# Main (terminal UI) loop:
query_sentence = str(input("Input sentence (q to quit): "))
while query_sentence!="q":
    query_operation_sequence = str(input("Input operation sequence: "))
    result, comparison = engine.process(query_sentence, query_operation_sequence)
    # Output
    print(f"Number of matched documents: {len(result)}")
    print(f"Minimum number of comparison required: {comparison}")
    print("List of retrieved document names: ")
    for listing in result: print(f'\t{listing}')
    query_sentence = str(input("\nInput sentence (q to quit): "))