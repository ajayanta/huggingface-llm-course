from transformers import pipeline

def sentiment_analysis_demo():
    print("=== Sentiment Analysis ===")
    classifier = pipeline("sentiment-analysis")
    print(classifier("I've been waiting for a HuggingFace course my whole life."))
    print(classifier(["I love this!", "I hate that."]))
    print()

def zero_shot_classification_demo():
    print("=== Zero-Shot Classification ===")
    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"]
    )
    print(result)
    print()

def text_generation_demo():
    print("=== Text Generation ===")
    generator = pipeline("text-generation")
    output = generator("In this course, we will teach you how to", max_length=30, num_return_sequences=2)
    print(output)
    print()

def fill_mask_demo():
    print("=== Fill Mask ===")
    unmasker = pipeline("fill-mask")
    results = unmasker("This course will teach you all about <mask> models.", top_k=2)
    for res in results:
        print(res)
    print()

def ner_demo():
    print("=== Named Entity Recognition (NER) ===")
    ner = pipeline("ner", grouped_entities=True)
    results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    for res in results:
        print(res)
    print()

def question_answering_demo():
    print("=== Question Answering ===")
    question_answerer = pipeline("question-answering")
    result = question_answerer(
        question="Where do I work?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )
    print(result)
    print()

def summarization_demo():
    print("=== Summarization ===")
    summarizer = pipeline("summarization")
    text = (
        "America has changed dramatically during recent years. Not only has the number of "
        "graduates in traditional engineering disciplines such as mechanical, civil, "
        "electrical, chemical, and aeronautical engineering declined, but in most of "
        "the premier American universities engineering curricula now concentrate on "
        "and encourage largely the study of engineering science. As a result, there "
        "are declining offerings in engineering subjects dealing with infrastructure, "
        "the environment, and related issues, and greater concentration on high "
        "technology subjects, largely supporting increasingly complex scientific "
        "developments. While the latter is important, it should not be at the expense "
        "of more traditional engineering."
    )
    summary = summarizer(text)
    print(summary)
    print()

def translation_demo():
    print("=== Translation (French to English) ===")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    result = translator("Ce cours est produit par Hugging Face.")
    print(result)
    print()

def main():
    sentiment_analysis_demo()
    zero_shot_classification_demo()
    text_generation_demo()
    fill_mask_demo()
    ner_demo()
    question_answering_demo()
    summarization_demo()
    translation_demo()

if __name__ == "__main__":
    main()
