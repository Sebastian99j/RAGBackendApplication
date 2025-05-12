# BLEU
from nltk.translate.bleu_score import sentence_bleu

reference = [["this", "is", "a", "test"]]
candidate = ["this", "is", "test"]

print("BLEU-1:", sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print("BLEU-2:", sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))

#ROGUE
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score("the cat was found", "the cat is found")

for key in scores:
    print(f"{key}: precision={scores[key].precision:.2f}, recall={scores[key].recall:.2f}, f1={scores[key].fmeasure:.2f}")

#TEST NA RZECZYWISTYM ZDANIU
# === Przygotowanie danych ===
reference_text = "Kot siedzi na dachu"
candidate_text = "Na dachu siedzi kot"

# Tokenizacja
reference_tokens = [reference_text.lower().split()]
candidate_tokens = candidate_text.lower().split()

# === BLEU ===
bleu1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0))
bleu2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0))

print(f"\n--- BLEU ---")
print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-2: {bleu2:.4f}")

# === ROUGE ===
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(reference_text, candidate_text)

print(f"\n--- ROUGE ---")
for key in rouge_scores:
    r = rouge_scores[key]
    print(f"{key}: precision={r.precision:.2f}, recall={r.recall:.2f}, f1={r.fmeasure:.2f}")