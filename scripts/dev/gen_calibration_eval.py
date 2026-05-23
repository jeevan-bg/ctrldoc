"""Generate the calibration_eval.jsonl starter dataset.

200 NLI cases balanced across {entailment, contradiction, neutral}
and spanning the six DOC_TYPES from claim_extraction. Premises and
hypotheses are authored as deterministic templates so the dataset
is reproducible and reviewable — this script is the source of
truth; tests/eval/calibration_eval.jsonl is its output.

Run from repo root:
    .venv/bin/python scripts/dev/gen_calibration_eval.py

SPEC-REF: §6.5 (probabilistic edges + calibration)
"""

from __future__ import annotations

import json
from pathlib import Path

# Each tuple: (premise, hypothesis, label, doc_type)
# Labels: "entailment" (hypothesis follows from premise),
#         "contradiction" (hypothesis denies premise),
#         "neutral" (premise neither entails nor contradicts).

CASES: list[tuple[str, str, str, str]] = []


# --- SPEC doc_type ---------------------------------------------------

SPEC_ENTAIL = [
    (
        "The system uses consistent hashing to distribute keys across exactly 256 virtual nodes per physical shard.",
        "Each shard hosts 256 virtual nodes.",
    ),
    (
        "Writes require quorum acknowledgement from a majority of replicas before the client receives a success response.",
        "A write is only acknowledged after a quorum of replicas confirm.",
    ),
    (
        "All API responses are serialized as JSON encoded in UTF-8 with no BOM character at the start.",
        "API responses use UTF-8 JSON encoding.",
    ),
    (
        "The cluster maintains three replicas per shard distributed across distinct availability zones for fault tolerance.",
        "Each shard has three replicas spread across availability zones.",
    ),
    (
        "Gossip messages are sent every 200 milliseconds between random pairs of nodes to propagate membership state.",
        "Membership state propagates via 200 ms gossip rounds.",
    ),
    (
        "Reads bypass the leader and may be served by any in-sync replica when the request specifies eventual consistency.",
        "Followers can serve eventually consistent reads.",
    ),
    (
        "The cache implements an LRU eviction policy with a maximum capacity of 10,000 entries per node.",
        "Cache entries are evicted in least-recently-used order.",
    ),
    (
        "Authentication tokens are bearer tokens transmitted in the Authorization header using the Bearer scheme.",
        "API authentication uses bearer tokens.",
    ),
    (
        "Each request must include an idempotency key as a UUID v4 string in the X-Idempotency-Key header.",
        "Clients send an idempotency key per request.",
    ),
    (
        "The system logs every write operation to a durable append-only journal flushed to disk before acknowledgement.",
        "Writes are journaled before being acknowledged.",
    ),
    (
        "Background compaction runs hourly and reclaims tombstoned space older than the configured grace period.",
        "Compaction reclaims tombstone space on an hourly cadence.",
    ),
    (
        "A health check endpoint at /healthz returns HTTP 200 when the node is accepting traffic.",
        "Healthy nodes return 200 from /healthz.",
    ),
]

SPEC_CONTRADICT = [
    (
        "The system uses consistent hashing to distribute keys across exactly 256 virtual nodes per physical shard.",
        "Keys are distributed using a round-robin scheme without hashing.",
    ),
    (
        "Writes require quorum acknowledgement from a majority of replicas before the client receives a success response.",
        "A single replica acknowledgement is sufficient for a successful write.",
    ),
    (
        "All API responses are serialized as JSON encoded in UTF-8 with no BOM character at the start.",
        "API responses are returned as XML documents.",
    ),
    (
        "The cluster maintains three replicas per shard distributed across distinct availability zones for fault tolerance.",
        "Each shard has only one replica in a single zone.",
    ),
    (
        "Gossip messages are sent every 200 milliseconds between random pairs of nodes to propagate membership state.",
        "Membership state is broadcast once per hour from a central coordinator.",
    ),
    (
        "Reads bypass the leader and may be served by any in-sync replica when the request specifies eventual consistency.",
        "All reads must be routed through the leader regardless of consistency level.",
    ),
    (
        "The cache implements an LRU eviction policy with a maximum capacity of 10,000 entries per node.",
        "Cache entries are never evicted and capacity is unbounded.",
    ),
    (
        "Authentication tokens are bearer tokens transmitted in the Authorization header using the Bearer scheme.",
        "Authentication is performed via mutual TLS without any tokens.",
    ),
    (
        "Each request must include an idempotency key as a UUID v4 string in the X-Idempotency-Key header.",
        "The API does not accept or process any idempotency keys.",
    ),
    (
        "The system logs every write operation to a durable append-only journal flushed to disk before acknowledgement.",
        "Write operations are never persisted to a journal.",
    ),
    (
        "Background compaction runs hourly and reclaims tombstoned space older than the configured grace period.",
        "Compaction has been removed; tombstones accumulate indefinitely.",
    ),
    (
        "A health check endpoint at /healthz returns HTTP 200 when the node is accepting traffic.",
        "The /healthz endpoint returns HTTP 500 for healthy nodes.",
    ),
]

SPEC_NEUTRAL = [
    (
        "The system uses consistent hashing to distribute keys across exactly 256 virtual nodes per physical shard.",
        "The implementation is written in Rust with zero external dependencies.",
    ),
    (
        "Writes require quorum acknowledgement from a majority of replicas before the client receives a success response.",
        "The default replication factor can be changed at runtime via an admin API.",
    ),
    (
        "All API responses are serialized as JSON encoded in UTF-8 with no BOM character at the start.",
        "The maximum payload size for a single response is 16 megabytes.",
    ),
    (
        "The cluster maintains three replicas per shard distributed across distinct availability zones for fault tolerance.",
        "Each replica costs approximately five dollars per month to operate.",
    ),
    (
        "Gossip messages are sent every 200 milliseconds between random pairs of nodes to propagate membership state.",
        "The cluster topology is visualised in the operator dashboard.",
    ),
    (
        "Reads bypass the leader and may be served by any in-sync replica when the request specifies eventual consistency.",
        "Read latency at the 99th percentile is under 50 milliseconds.",
    ),
    (
        "The cache implements an LRU eviction policy with a maximum capacity of 10,000 entries per node.",
        "Cache statistics are exposed via a Prometheus metrics endpoint.",
    ),
    (
        "Authentication tokens are bearer tokens transmitted in the Authorization header using the Bearer scheme.",
        "Tokens are issued by an external identity provider over OIDC.",
    ),
    (
        "Each request must include an idempotency key as a UUID v4 string in the X-Idempotency-Key header.",
        "Rate limits are enforced per API key at 1000 requests per minute.",
    ),
    (
        "The system logs every write operation to a durable append-only journal flushed to disk before acknowledgement.",
        "Disk capacity per node is provisioned at 10 terabytes.",
    ),
    (
        "Background compaction runs hourly and reclaims tombstoned space older than the configured grace period.",
        "Compaction throughput is bounded by available network bandwidth.",
    ),
    (
        "A health check endpoint at /healthz returns HTTP 200 when the node is accepting traffic.",
        "The operator can drain a node manually via the CLI.",
    ),
]


# --- LEGAL doc_type --------------------------------------------------

LEGAL_ENTAIL = [
    (
        "The data controller shall retain personal data for no longer than is necessary for the purposes for which it was collected.",
        "Personal data must not be kept beyond the necessary retention period.",
    ),
    (
        "All disputes arising under this agreement shall be resolved exclusively through binding arbitration in the State of New York.",
        "Disputes are settled through arbitration in New York.",
    ),
    (
        "Each party agrees to indemnify the other against claims brought by third parties arising from gross negligence.",
        "Parties indemnify one another for third-party gross-negligence claims.",
    ),
    (
        "The licensee may not sublicense, sell, lease, or otherwise transfer the software to any third party without prior written consent.",
        "Sublicensing the software requires prior written consent.",
    ),
    (
        "Notices required under this agreement shall be delivered by certified mail with return receipt requested.",
        "Required notices must be sent by certified mail.",
    ),
    (
        "The supplier warrants that the goods will conform to the specifications set forth in Schedule A for a period of twelve months.",
        "Goods carry a twelve-month warranty against the Schedule A specifications.",
    ),
    (
        "Confidential Information shall remain the sole property of the Disclosing Party and no license is granted by its disclosure.",
        "Disclosure of Confidential Information does not grant a license.",
    ),
    (
        "Either party may terminate this agreement upon thirty days written notice to the other party.",
        "Termination requires thirty days written notice.",
    ),
    (
        "The contractor shall be paid within thirty days of receipt of an undisputed invoice.",
        "Undisputed invoices must be paid within thirty days.",
    ),
    (
        "Force majeure includes any event beyond the reasonable control of a party including war, fire, flood, and pandemic.",
        "A pandemic qualifies as a force majeure event.",
    ),
    (
        "All employees shall complete annual data protection training within ninety days of their hire date.",
        "New employees must complete data protection training in their first ninety days.",
    ),
    (
        "The processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.",
        "Processors are required to implement risk-appropriate security measures.",
    ),
]

LEGAL_CONTRADICT = [
    (
        "The data controller shall retain personal data for no longer than is necessary for the purposes for which it was collected.",
        "Personal data may be retained indefinitely after collection.",
    ),
    (
        "All disputes arising under this agreement shall be resolved exclusively through binding arbitration in the State of New York.",
        "Disputes must be litigated in the courts of California.",
    ),
    (
        "Each party agrees to indemnify the other against claims brought by third parties arising from gross negligence.",
        "No party is obligated to indemnify the other under any circumstances.",
    ),
    (
        "The licensee may not sublicense, sell, lease, or otherwise transfer the software to any third party without prior written consent.",
        "The licensee may freely sublicense the software to anyone.",
    ),
    (
        "Notices required under this agreement shall be delivered by certified mail with return receipt requested.",
        "Notices may only be delivered via informal email.",
    ),
    (
        "The supplier warrants that the goods will conform to the specifications set forth in Schedule A for a period of twelve months.",
        "The supplier disclaims any warranty on the goods.",
    ),
    (
        "Confidential Information shall remain the sole property of the Disclosing Party and no license is granted by its disclosure.",
        "Disclosing Confidential Information transfers ownership to the Receiving Party.",
    ),
    (
        "Either party may terminate this agreement upon thirty days written notice to the other party.",
        "Neither party may terminate this agreement before the five-year term ends.",
    ),
    (
        "The contractor shall be paid within thirty days of receipt of an undisputed invoice.",
        "Invoice payment is due within one hundred eighty days.",
    ),
    (
        "Force majeure includes any event beyond the reasonable control of a party including war, fire, flood, and pandemic.",
        "Pandemics are explicitly excluded from the force majeure clause.",
    ),
    (
        "All employees shall complete annual data protection training within ninety days of their hire date.",
        "Data protection training is optional and never required.",
    ),
    (
        "The processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.",
        "The processor is not required to implement any security measures.",
    ),
]

LEGAL_NEUTRAL = [
    (
        "The data controller shall retain personal data for no longer than is necessary for the purposes for which it was collected.",
        "The data controller's headquarters are located in Dublin.",
    ),
    (
        "All disputes arising under this agreement shall be resolved exclusively through binding arbitration in the State of New York.",
        "The agreement was executed on the fifteenth day of March.",
    ),
    (
        "Each party agrees to indemnify the other against claims brought by third parties arising from gross negligence.",
        "Both parties have purchased liability insurance from the same carrier.",
    ),
    (
        "The licensee may not sublicense, sell, lease, or otherwise transfer the software to any third party without prior written consent.",
        "The software is distributed in compiled binary form only.",
    ),
    (
        "Notices required under this agreement shall be delivered by certified mail with return receipt requested.",
        "The parties communicate primarily through a shared Slack workspace.",
    ),
    (
        "The supplier warrants that the goods will conform to the specifications set forth in Schedule A for a period of twelve months.",
        "Schedule A was drafted by the engineering team at the buyer.",
    ),
    (
        "Confidential Information shall remain the sole property of the Disclosing Party and no license is granted by its disclosure.",
        "The Disclosing Party publishes an annual transparency report.",
    ),
    (
        "Either party may terminate this agreement upon thirty days written notice to the other party.",
        "The agreement was signed in three identical originals.",
    ),
    (
        "The contractor shall be paid within thirty days of receipt of an undisputed invoice.",
        "Invoices are submitted via the buyer's procurement portal.",
    ),
    (
        "Force majeure includes any event beyond the reasonable control of a party including war, fire, flood, and pandemic.",
        "The parties review the force majeure provisions every two years.",
    ),
    (
        "All employees shall complete annual data protection training within ninety days of their hire date.",
        "The training is delivered through an online learning platform.",
    ),
    (
        "The processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.",
        "The processor is headquartered in a separate jurisdiction from the controller.",
    ),
]


# --- ACADEMIC doc_type -----------------------------------------------

ACADEMIC_ENTAIL = [
    (
        "We trained a transformer model with 12 layers and 768 hidden dimensions on a corpus of 100 billion tokens.",
        "The model has 12 transformer layers.",
    ),
    (
        "Our experiments show that the proposed method outperforms the baseline by 4.2 BLEU points on the WMT 2014 English-German test set.",
        "The proposed method beats the baseline on WMT 2014 English-German.",
    ),
    (
        "The dataset consists of 50,000 image-caption pairs split into 40,000 training examples and 10,000 held-out test examples.",
        "There are 10,000 test images in the dataset.",
    ),
    (
        "We use Adam optimizer with learning rate 1e-4 and cosine annealing for the first 10,000 training steps.",
        "Adam is the optimizer in our training setup.",
    ),
    (
        "All experiments were repeated five times with different random seeds and we report the mean and standard deviation.",
        "Each experiment was run with five different seeds.",
    ),
    (
        "The participants were 240 undergraduate students recruited from an introductory psychology course in exchange for course credit.",
        "Undergraduates received course credit for participating.",
    ),
    (
        "Our null hypothesis was rejected at the p < 0.01 significance level using a two-tailed paired t-test.",
        "The null hypothesis was rejected at the 1 percent significance level.",
    ),
    (
        "We pre-trained on the C4 corpus for one epoch and fine-tuned on each downstream task for three epochs.",
        "Fine-tuning lasted three epochs per task.",
    ),
    (
        "The proposed loss function adds an L2 regularisation term with coefficient 0.01 to the standard cross-entropy.",
        "An L2 regularisation term is added to the cross-entropy loss.",
    ),
    (
        "Inter-annotator agreement measured by Cohen's kappa was 0.78, indicating substantial agreement.",
        "Cohen's kappa for the annotators was 0.78.",
    ),
    (
        "We report results on three benchmarks: GLUE, SuperGLUE, and SQuAD 2.0.",
        "SQuAD 2.0 is one of the benchmarks evaluated.",
    ),
    (
        "The model was evaluated on a held-out test set never seen during training or validation.",
        "The test set was held out from training.",
    ),
]

ACADEMIC_CONTRADICT = [
    (
        "We trained a transformer model with 12 layers and 768 hidden dimensions on a corpus of 100 billion tokens.",
        "The model is a recurrent neural network with no attention mechanism.",
    ),
    (
        "Our experiments show that the proposed method outperforms the baseline by 4.2 BLEU points on the WMT 2014 English-German test set.",
        "The proposed method performs worse than the baseline on WMT 2014.",
    ),
    (
        "The dataset consists of 50,000 image-caption pairs split into 40,000 training examples and 10,000 held-out test examples.",
        "The dataset has no held-out test split at all.",
    ),
    (
        "We use Adam optimizer with learning rate 1e-4 and cosine annealing for the first 10,000 training steps.",
        "We use stochastic gradient descent with a fixed learning rate.",
    ),
    (
        "All experiments were repeated five times with different random seeds and we report the mean and standard deviation.",
        "Each experiment was run only once with a single fixed seed.",
    ),
    (
        "The participants were 240 undergraduate students recruited from an introductory psychology course in exchange for course credit.",
        "The participants were professional clinicians paid a market rate.",
    ),
    (
        "Our null hypothesis was rejected at the p < 0.01 significance level using a two-tailed paired t-test.",
        "We failed to reject the null hypothesis at any conventional level.",
    ),
    (
        "We pre-trained on the C4 corpus for one epoch and fine-tuned on each downstream task for three epochs.",
        "No pre-training was performed; the model was trained from scratch on each task.",
    ),
    (
        "The proposed loss function adds an L2 regularisation term with coefficient 0.01 to the standard cross-entropy.",
        "The loss function uses no regularisation term whatsoever.",
    ),
    (
        "Inter-annotator agreement measured by Cohen's kappa was 0.78, indicating substantial agreement.",
        "Cohen's kappa was 0.05, indicating no better than chance agreement.",
    ),
    (
        "We report results on three benchmarks: GLUE, SuperGLUE, and SQuAD 2.0.",
        "We do not evaluate on any benchmark beyond our internal dataset.",
    ),
    (
        "The model was evaluated on a held-out test set never seen during training or validation.",
        "The test set was included in the training data.",
    ),
]

ACADEMIC_NEUTRAL = [
    (
        "We trained a transformer model with 12 layers and 768 hidden dimensions on a corpus of 100 billion tokens.",
        "The first author of the paper is affiliated with Stanford University.",
    ),
    (
        "Our experiments show that the proposed method outperforms the baseline by 4.2 BLEU points on the WMT 2014 English-German test set.",
        "The proposed method also reduces inference latency by 30 percent.",
    ),
    (
        "The dataset consists of 50,000 image-caption pairs split into 40,000 training examples and 10,000 held-out test examples.",
        "The dataset will be released publicly under a CC-BY licence.",
    ),
    (
        "We use Adam optimizer with learning rate 1e-4 and cosine annealing for the first 10,000 training steps.",
        "Training was performed on a cluster of eight A100 GPUs.",
    ),
    (
        "All experiments were repeated five times with different random seeds and we report the mean and standard deviation.",
        "Hyperparameters were tuned on a separate validation split.",
    ),
    (
        "The participants were 240 undergraduate students recruited from an introductory psychology course in exchange for course credit.",
        "The study was approved by the institutional review board prior to data collection.",
    ),
    (
        "Our null hypothesis was rejected at the p < 0.01 significance level using a two-tailed paired t-test.",
        "Effect sizes were reported alongside the significance tests.",
    ),
    (
        "We pre-trained on the C4 corpus for one epoch and fine-tuned on each downstream task for three epochs.",
        "The C4 corpus is publicly available from a major cloud provider.",
    ),
    (
        "The proposed loss function adds an L2 regularisation term with coefficient 0.01 to the standard cross-entropy.",
        "The model was implemented in PyTorch using mixed-precision training.",
    ),
    (
        "Inter-annotator agreement measured by Cohen's kappa was 0.78, indicating substantial agreement.",
        "The annotation interface was a custom web application.",
    ),
    (
        "We report results on three benchmarks: GLUE, SuperGLUE, and SQuAD 2.0.",
        "Reproducibility scripts will be released on GitHub upon publication.",
    ),
    (
        "The model was evaluated on a held-out test set never seen during training or validation.",
        "Model checkpoints are available on the Hugging Face hub.",
    ),
]


# --- EDUCATIONAL doc_type --------------------------------------------

EDU_ENTAIL = [
    (
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight.",
        "Glucose is one of the products of photosynthesis.",
    ),
    (
        "The mitochondria are the organelles responsible for producing ATP through cellular respiration in eukaryotic cells.",
        "Mitochondria produce ATP in eukaryotic cells.",
    ),
    (
        "Sound waves are longitudinal pressure variations that propagate through a medium such as air or water.",
        "Sound waves require a medium to travel through.",
    ),
    (
        "Mercury is the closest planet to the Sun and completes one orbit every 88 Earth days.",
        "Mercury's orbital period is 88 Earth days.",
    ),
    (
        "An atom consists of a nucleus containing protons and neutrons surrounded by electrons in orbital shells.",
        "Electrons orbit the nucleus of an atom.",
    ),
    (
        "The Pythagorean theorem states that in any right triangle the square of the hypotenuse equals the sum of the squares of the other two sides.",
        "The hypotenuse squared equals the sum of the squares of the legs in a right triangle.",
    ),
    (
        "In the water cycle, evaporation lifts water vapor into the atmosphere where it condenses into clouds and eventually falls as precipitation.",
        "Precipitation is part of the water cycle.",
    ),
    (
        "The French Revolution began in 1789 with the storming of the Bastille and led to the abolition of the monarchy in France.",
        "The Bastille was stormed at the start of the French Revolution.",
    ),
    (
        "Plants absorb water through their roots and transport it upward through specialized tissue called xylem.",
        "Xylem carries water from the roots upward in plants.",
    ),
    (
        "Newton's first law of motion states that an object at rest stays at rest unless acted upon by an external force.",
        "An object at rest remains at rest absent an external force.",
    ),
    (
        "The human heart has four chambers: two atria on top and two ventricles on the bottom.",
        "The heart contains four chambers.",
    ),
    (
        "Earth completes one rotation on its axis approximately every 24 hours, which gives us the cycle of day and night.",
        "Earth's daily rotation produces day and night.",
    ),
]

EDU_CONTRADICT = [
    (
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight.",
        "Photosynthesis produces no oxygen at all.",
    ),
    (
        "The mitochondria are the organelles responsible for producing ATP through cellular respiration in eukaryotic cells.",
        "ATP is produced exclusively in the cell nucleus, not the mitochondria.",
    ),
    (
        "Sound waves are longitudinal pressure variations that propagate through a medium such as air or water.",
        "Sound waves travel through a perfect vacuum without difficulty.",
    ),
    (
        "Mercury is the closest planet to the Sun and completes one orbit every 88 Earth days.",
        "Mercury takes one Earth year to orbit the Sun.",
    ),
    (
        "An atom consists of a nucleus containing protons and neutrons surrounded by electrons in orbital shells.",
        "Atoms have no nucleus and consist only of free-floating electrons.",
    ),
    (
        "The Pythagorean theorem states that in any right triangle the square of the hypotenuse equals the sum of the squares of the other two sides.",
        "The Pythagorean theorem applies equally to all triangles regardless of angle.",
    ),
    (
        "In the water cycle, evaporation lifts water vapor into the atmosphere where it condenses into clouds and eventually falls as precipitation.",
        "Water vapor in the atmosphere never returns to the surface as precipitation.",
    ),
    (
        "The French Revolution began in 1789 with the storming of the Bastille and led to the abolition of the monarchy in France.",
        "The French Revolution restored the French monarchy and abolished the republic.",
    ),
    (
        "Plants absorb water through their roots and transport it upward through specialized tissue called xylem.",
        "Plants absorb water through their leaves and transport it downward to the roots.",
    ),
    (
        "Newton's first law of motion states that an object at rest stays at rest unless acted upon by an external force.",
        "An object at rest will spontaneously begin moving without any external force.",
    ),
    (
        "The human heart has four chambers: two atria on top and two ventricles on the bottom.",
        "The human heart consists of a single chamber with no internal divisions.",
    ),
    (
        "Earth completes one rotation on its axis approximately every 24 hours, which gives us the cycle of day and night.",
        "Earth does not rotate on its axis at all.",
    ),
]

EDU_NEUTRAL = [
    (
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight.",
        "Many crops require irrigation in addition to rainfall.",
    ),
    (
        "The mitochondria are the organelles responsible for producing ATP through cellular respiration in eukaryotic cells.",
        "Mitochondria are believed to have evolved from ancient symbiotic bacteria.",
    ),
    (
        "Sound waves are longitudinal pressure variations that propagate through a medium such as air or water.",
        "Concert halls are designed with attention to acoustic reflection.",
    ),
    (
        "Mercury is the closest planet to the Sun and completes one orbit every 88 Earth days.",
        "Mercury has been visited by spacecraft from NASA.",
    ),
    (
        "An atom consists of a nucleus containing protons and neutrons surrounded by electrons in orbital shells.",
        "The atomic theory was significantly advanced in the early twentieth century.",
    ),
    (
        "The Pythagorean theorem states that in any right triangle the square of the hypotenuse equals the sum of the squares of the other two sides.",
        "Pythagoras founded a school of philosophy in ancient Greece.",
    ),
    (
        "In the water cycle, evaporation lifts water vapor into the atmosphere where it condenses into clouds and eventually falls as precipitation.",
        "Meteorologists track weather patterns using radar and satellite data.",
    ),
    (
        "The French Revolution began in 1789 with the storming of the Bastille and led to the abolition of the monarchy in France.",
        "The Eiffel Tower was constructed for the World's Fair in 1889.",
    ),
    (
        "Plants absorb water through their roots and transport it upward through specialized tissue called xylem.",
        "Different plant species require different amounts of sunlight to thrive.",
    ),
    (
        "Newton's first law of motion states that an object at rest stays at rest unless acted upon by an external force.",
        "Newton also made significant contributions to optics and calculus.",
    ),
    (
        "The human heart has four chambers: two atria on top and two ventricles on the bottom.",
        "Cardiologists specialize in diagnosing heart conditions.",
    ),
    (
        "Earth completes one rotation on its axis approximately every 24 hours, which gives us the cycle of day and night.",
        "Earth is the third planet from the Sun in our solar system.",
    ),
]


# --- NARRATIVE doc_type ----------------------------------------------

NARR_ENTAIL = [
    (
        "After the long voyage Maria finally arrived in Lisbon at dawn, exhausted but eager to see her grandmother.",
        "Maria arrived in Lisbon after a long journey.",
    ),
    (
        "The detective glanced at the rain-streaked window and decided that the suspect must have been bluffing about the alibi.",
        "The detective doubted the suspect's alibi.",
    ),
    (
        "When the storm finally passed, the village inspected the damage and discovered that the old bridge had been swept away.",
        "The storm destroyed the old bridge.",
    ),
    (
        "Although her teacher had warned her against it, Anya volunteered to give the closing speech at the ceremony.",
        "Anya volunteered for the closing speech despite her teacher's warning.",
    ),
    (
        "The general dispatched two regiments to reinforce the eastern flank where the enemy attack had been heaviest all morning.",
        "Reinforcements were sent to the eastern flank.",
    ),
    (
        "Carlos walked into the kitchen, opened the refrigerator, and grimaced at the smell of last week's leftovers.",
        "Carlos noticed an unpleasant smell in his refrigerator.",
    ),
    (
        "The merchant's daughter refused the prince's offer of marriage despite her family's increasingly desperate financial situation.",
        "The merchant's daughter turned down the prince.",
    ),
    (
        "After three years of failed harvests, the villagers gathered to plan the move south to fertile lands they had only heard about in stories.",
        "The villagers decided to migrate south.",
    ),
    (
        "Jamal finished the marathon in just under three hours, beating his personal best by nearly four minutes.",
        "Jamal set a new personal best in the marathon.",
    ),
    (
        "The captain ordered the lifeboats to be lowered the moment the second explosion rocked the lower decks.",
        "The captain called for evacuation after the second explosion.",
    ),
    (
        "Yuki had been studying Spanish in secret for two years before she surprised her parents by switching to it at dinner.",
        "Yuki secretly studied Spanish for two years before revealing it.",
    ),
    (
        "When the message finally reached the outpost, the soldiers had already begun rationing their last barrel of water.",
        "Water was already being rationed when the message arrived.",
    ),
]

NARR_CONTRADICT = [
    (
        "After the long voyage Maria finally arrived in Lisbon at dawn, exhausted but eager to see her grandmother.",
        "Maria never left her home town and refused to travel anywhere.",
    ),
    (
        "The detective glanced at the rain-streaked window and decided that the suspect must have been bluffing about the alibi.",
        "The detective fully believed the suspect's account.",
    ),
    (
        "When the storm finally passed, the village inspected the damage and discovered that the old bridge had been swept away.",
        "The storm left the village entirely untouched.",
    ),
    (
        "Although her teacher had warned her against it, Anya volunteered to give the closing speech at the ceremony.",
        "Anya skipped the ceremony and refused to speak at all.",
    ),
    (
        "The general dispatched two regiments to reinforce the eastern flank where the enemy attack had been heaviest all morning.",
        "The general ordered a full retreat from the entire eastern theater.",
    ),
    (
        "Carlos walked into the kitchen, opened the refrigerator, and grimaced at the smell of last week's leftovers.",
        "Carlos found the refrigerator to be empty and spotless.",
    ),
    (
        "The merchant's daughter refused the prince's offer of marriage despite her family's increasingly desperate financial situation.",
        "The merchant's daughter eagerly accepted the prince's marriage proposal.",
    ),
    (
        "After three years of failed harvests, the villagers gathered to plan the move south to fertile lands they had only heard about in stories.",
        "The villagers decided to remain in their homeland and trust the next harvest.",
    ),
    (
        "Jamal finished the marathon in just under three hours, beating his personal best by nearly four minutes.",
        "Jamal dropped out of the marathon before reaching the halfway point.",
    ),
    (
        "The captain ordered the lifeboats to be lowered the moment the second explosion rocked the lower decks.",
        "The captain refused to deploy any lifeboats regardless of the danger.",
    ),
    (
        "Yuki had been studying Spanish in secret for two years before she surprised her parents by switching to it at dinner.",
        "Yuki had never shown any interest in learning a foreign language.",
    ),
    (
        "When the message finally reached the outpost, the soldiers had already begun rationing their last barrel of water.",
        "The outpost was abundantly supplied and water rationing had never been considered.",
    ),
]

NARR_NEUTRAL = [
    (
        "After the long voyage Maria finally arrived in Lisbon at dawn, exhausted but eager to see her grandmother.",
        "Maria's grandmother had spent her career as a botanist.",
    ),
    (
        "The detective glanced at the rain-streaked window and decided that the suspect must have been bluffing about the alibi.",
        "The detective had recently been promoted after solving a string of difficult cases.",
    ),
    (
        "When the storm finally passed, the village inspected the damage and discovered that the old bridge had been swept away.",
        "The village had been founded by traders three hundred years earlier.",
    ),
    (
        "Although her teacher had warned her against it, Anya volunteered to give the closing speech at the ceremony.",
        "The ceremony was held in the school's main auditorium.",
    ),
    (
        "The general dispatched two regiments to reinforce the eastern flank where the enemy attack had been heaviest all morning.",
        "The general had served in three previous campaigns under a different command.",
    ),
    (
        "Carlos walked into the kitchen, opened the refrigerator, and grimaced at the smell of last week's leftovers.",
        "Carlos lived in a small apartment near the train station.",
    ),
    (
        "The merchant's daughter refused the prince's offer of marriage despite her family's increasingly desperate financial situation.",
        "The kingdom had been at peace for nearly two decades.",
    ),
    (
        "After three years of failed harvests, the villagers gathered to plan the move south to fertile lands they had only heard about in stories.",
        "The villagers traded woven baskets at the regional market each spring.",
    ),
    (
        "Jamal finished the marathon in just under three hours, beating his personal best by nearly four minutes.",
        "Jamal had taken up running after recovering from a knee injury.",
    ),
    (
        "The captain ordered the lifeboats to be lowered the moment the second explosion rocked the lower decks.",
        "The ship had been built in a Glasgow shipyard twelve years earlier.",
    ),
    (
        "Yuki had been studying Spanish in secret for two years before she surprised her parents by switching to it at dinner.",
        "Yuki's family lived in a quiet suburb on the northern edge of the city.",
    ),
    (
        "When the message finally reached the outpost, the soldiers had already begun rationing their last barrel of water.",
        "The outpost had been established near a now-dry riverbed.",
    ),
]


# --- TECHNICAL doc_type ----------------------------------------------

TECH_ENTAIL = [
    (
        "The build pipeline runs unit tests, integration tests, and a security scan before publishing the container image to the registry.",
        "The pipeline executes three test stages before publishing.",
    ),
    (
        "To create a new project, run the init command from the repository root which scaffolds the directory and installs dependencies.",
        "The init command must be run at the repository root.",
    ),
    (
        "All log entries are written in JSON format with ISO 8601 timestamps and a correlation ID for distributed tracing.",
        "Logs use ISO 8601 timestamps.",
    ),
    (
        "The library exposes a synchronous API as well as an async variant under the asyncio submodule for non-blocking workflows.",
        "An asynchronous API is available in addition to the synchronous one.",
    ),
    (
        "Connection pooling is enabled by default with a maximum of 20 connections per host and a 30 second idle timeout.",
        "The connection pool's default maximum is 20 per host.",
    ),
    (
        "The migrate command must be run before starting the server whenever a new database schema version is released.",
        "Running migrate is a prerequisite for starting the server after schema upgrades.",
    ),
    (
        "Configuration values are loaded first from environment variables and then from the optional config.yaml file in the working directory.",
        "Environment variables take precedence over the config.yaml file.",
    ),
    (
        "The CLI ships with shell completions for bash, zsh, and fish that can be installed via the install-completions subcommand.",
        "Shell completion installation is handled by a dedicated subcommand.",
    ),
    (
        "Webhook payloads are signed using HMAC SHA-256 with a secret shared between the sender and receiver.",
        "Webhook payloads carry an HMAC SHA-256 signature.",
    ),
    (
        "The default retry policy uses exponential backoff with a base delay of 100 milliseconds and a maximum of five attempts.",
        "Retries are capped at five attempts by default.",
    ),
    (
        "Plugins are discovered at startup by scanning the plugins directory and loading any file ending in plugin.py.",
        "Plugin discovery scans the plugins directory at startup.",
    ),
    (
        "The dashboard refreshes its metrics every 15 seconds by polling the metrics endpoint over HTTPS.",
        "The dashboard polls metrics on a 15 second interval.",
    ),
]

TECH_CONTRADICT = [
    (
        "The build pipeline runs unit tests, integration tests, and a security scan before publishing the container image to the registry.",
        "The pipeline publishes the image directly without running any tests.",
    ),
    (
        "To create a new project, run the init command from the repository root which scaffolds the directory and installs dependencies.",
        "The init command can only be run from outside any repository.",
    ),
    (
        "All log entries are written in JSON format with ISO 8601 timestamps and a correlation ID for distributed tracing.",
        "Log entries are written in plain text with no timestamps at all.",
    ),
    (
        "The library exposes a synchronous API as well as an async variant under the asyncio submodule for non-blocking workflows.",
        "The library provides only a synchronous interface with no async support.",
    ),
    (
        "Connection pooling is enabled by default with a maximum of 20 connections per host and a 30 second idle timeout.",
        "Connection pooling is disabled by default and must be explicitly turned on.",
    ),
    (
        "The migrate command must be run before starting the server whenever a new database schema version is released.",
        "Schema migrations are applied silently by the server at startup without any user action.",
    ),
    (
        "Configuration values are loaded first from environment variables and then from the optional config.yaml file in the working directory.",
        "Configuration is loaded only from a hard-coded path; environment variables are ignored.",
    ),
    (
        "The CLI ships with shell completions for bash, zsh, and fish that can be installed via the install-completions subcommand.",
        "The CLI has no shell completion support of any kind.",
    ),
    (
        "Webhook payloads are signed using HMAC SHA-256 with a secret shared between the sender and receiver.",
        "Webhook payloads are sent unsigned and unauthenticated.",
    ),
    (
        "The default retry policy uses exponential backoff with a base delay of 100 milliseconds and a maximum of five attempts.",
        "There is no retry mechanism and any failure is fatal.",
    ),
    (
        "Plugins are discovered at startup by scanning the plugins directory and loading any file ending in plugin.py.",
        "Plugins are explicitly enumerated in a configuration file with no directory scanning.",
    ),
    (
        "The dashboard refreshes its metrics every 15 seconds by polling the metrics endpoint over HTTPS.",
        "The dashboard updates only when the user manually clicks the refresh button.",
    ),
]

TECH_NEUTRAL = [
    (
        "The build pipeline runs unit tests, integration tests, and a security scan before publishing the container image to the registry.",
        "The registry is hosted on a separate cloud account for blast-radius isolation.",
    ),
    (
        "To create a new project, run the init command from the repository root which scaffolds the directory and installs dependencies.",
        "Most users prefer to invoke the CLI via a shell alias.",
    ),
    (
        "All log entries are written in JSON format with ISO 8601 timestamps and a correlation ID for distributed tracing.",
        "Logs are shipped to a centralised observability platform for indexing.",
    ),
    (
        "The library exposes a synchronous API as well as an async variant under the asyncio submodule for non-blocking workflows.",
        "The library is distributed via the Python Package Index.",
    ),
    (
        "Connection pooling is enabled by default with a maximum of 20 connections per host and a 30 second idle timeout.",
        "TLS 1.3 is required for all outgoing connections.",
    ),
    (
        "The migrate command must be run before starting the server whenever a new database schema version is released.",
        "The database driver is bundled with the server binary.",
    ),
    (
        "Configuration values are loaded first from environment variables and then from the optional config.yaml file in the working directory.",
        "Secrets should be managed through a dedicated vault service.",
    ),
    (
        "The CLI ships with shell completions for bash, zsh, and fish that can be installed via the install-completions subcommand.",
        "The CLI binary is statically linked and runs on Linux, macOS, and Windows.",
    ),
    (
        "Webhook payloads are signed using HMAC SHA-256 with a secret shared between the sender and receiver.",
        "Webhook delivery is monitored via a dedicated health dashboard.",
    ),
    (
        "The default retry policy uses exponential backoff with a base delay of 100 milliseconds and a maximum of five attempts.",
        "Most production deployments override the retry policy via configuration.",
    ),
    (
        "Plugins are discovered at startup by scanning the plugins directory and loading any file ending in plugin.py.",
        "The plugin developer guide is published on the project website.",
    ),
    (
        "The dashboard refreshes its metrics every 15 seconds by polling the metrics endpoint over HTTPS.",
        "The dashboard front-end is a single-page application built with TypeScript.",
    ),
]


# --- BLOG doc_type (unused; retained for future expansion to a 7th type) ---

BLOG_ENTAIL = [
    (
        "After six months of remote work I have settled into a daily rhythm of deep focus blocks separated by short walks outside.",
        "The author follows a routine that alternates focused work with outdoor breaks.",
    ),
    (
        "Switching from coffee to green tea last quarter eliminated the mid-afternoon energy crashes I used to suffer.",
        "Green tea replaced coffee in the author's diet.",
    ),
    (
        "I recently read a book on stoic philosophy and found the chapter on amor fati especially relevant to dealing with setbacks.",
        "The author found one chapter of the stoic book particularly useful for setbacks.",
    ),
    (
        "Our family adopted a rescue dog from the local shelter last spring and the house has been noticeably more cheerful ever since.",
        "Adopting the rescue dog improved the household mood.",
    ),
    (
        "I have started keeping a brief journal each evening which has helped me notice patterns in my mood that I would otherwise have missed.",
        "Evening journaling helps the author track mood patterns.",
    ),
    (
        "After a frustrating year of failed launches, the team finally shipped a product that exceeded all of our internal projections.",
        "The team's recent product launch beat their projections.",
    ),
    (
        "Moving to a smaller town has reduced my commute from an hour each way down to under fifteen minutes door to door.",
        "Relocating shortened the author's commute significantly.",
    ),
    (
        "I have been cooking more meals at home this year and noticed that my grocery bill is actually lower than my old takeout spending.",
        "Home cooking has reduced the author's food spending compared to takeout.",
    ),
    (
        "The local farmer's market reopens every Saturday from May through October and is a favourite weekend stop for our family.",
        "The market operates on Saturdays during the warmer half of the year.",
    ),
    (
        "Reading thirty minutes before bed instead of scrolling on my phone has dramatically improved both my sleep quality and my mood.",
        "Substituting reading for phone use before bed improved the author's sleep and mood.",
    ),
    (
        "I built a small treehouse in the backyard last summer using lumber salvaged from a neighbour's barn renovation.",
        "Salvaged lumber went into the backyard treehouse.",
    ),
    (
        "Our weekly board game night with the neighbours has become the social highlight of the week for everyone involved.",
        "Board game night is the weekly social highlight for the participants.",
    ),
]

BLOG_CONTRADICT = [
    (
        "After six months of remote work I have settled into a daily rhythm of deep focus blocks separated by short walks outside.",
        "The author works in a chaotic schedule with no consistent routine.",
    ),
    (
        "Switching from coffee to green tea last quarter eliminated the mid-afternoon energy crashes I used to suffer.",
        "The author continues to drink coffee and experiences worse energy crashes than before.",
    ),
    (
        "I recently read a book on stoic philosophy and found the chapter on amor fati especially relevant to dealing with setbacks.",
        "The author abandoned the stoic philosophy book without reading any of its chapters.",
    ),
    (
        "Our family adopted a rescue dog from the local shelter last spring and the house has been noticeably more cheerful ever since.",
        "The family returned the dog within a week and remained without any pet.",
    ),
    (
        "I have started keeping a brief journal each evening which has helped me notice patterns in my mood that I would otherwise have missed.",
        "The author has never written in a journal and refuses to start one.",
    ),
    (
        "After a frustrating year of failed launches, the team finally shipped a product that exceeded all of our internal projections.",
        "The latest launch was the team's biggest failure to date.",
    ),
    (
        "Moving to a smaller town has reduced my commute from an hour each way down to under fifteen minutes door to door.",
        "Relocating made the author's commute significantly longer.",
    ),
    (
        "I have been cooking more meals at home this year and noticed that my grocery bill is actually lower than my old takeout spending.",
        "The author ordered takeout for every meal this year.",
    ),
    (
        "The local farmer's market reopens every Saturday from May through October and is a favourite weekend stop for our family.",
        "The farmer's market shut down permanently several years ago.",
    ),
    (
        "Reading thirty minutes before bed instead of scrolling on my phone has dramatically improved both my sleep quality and my mood.",
        "Scrolling on the phone before bed has been kept exactly the same with no change.",
    ),
    (
        "I built a small treehouse in the backyard last summer using lumber salvaged from a neighbour's barn renovation.",
        "The author has never built anything and lacks any tools.",
    ),
    (
        "Our weekly board game night with the neighbours has become the social highlight of the week for everyone involved.",
        "The board game tradition was abandoned years ago and never reestablished.",
    ),
]

BLOG_NEUTRAL = [
    (
        "After six months of remote work I have settled into a daily rhythm of deep focus blocks separated by short walks outside.",
        "The author keeps a small herb garden on the kitchen windowsill.",
    ),
    (
        "Switching from coffee to green tea last quarter eliminated the mid-afternoon energy crashes I used to suffer.",
        "The author has been considering taking up cycling this summer.",
    ),
    (
        "I recently read a book on stoic philosophy and found the chapter on amor fati especially relevant to dealing with setbacks.",
        "The author also enjoys reading historical fiction in the evenings.",
    ),
    (
        "Our family adopted a rescue dog from the local shelter last spring and the house has been noticeably more cheerful ever since.",
        "The family vacation last year was to a small coastal town.",
    ),
    (
        "I have started keeping a brief journal each evening which has helped me notice patterns in my mood that I would otherwise have missed.",
        "The author's favourite pen is a refillable fountain pen from Japan.",
    ),
    (
        "After a frustrating year of failed launches, the team finally shipped a product that exceeded all of our internal projections.",
        "The team is planning a small offsite to celebrate later this quarter.",
    ),
    (
        "Moving to a smaller town has reduced my commute from an hour each way down to under fifteen minutes door to door.",
        "The new house has a wood-burning fireplace in the living room.",
    ),
    (
        "I have been cooking more meals at home this year and noticed that my grocery bill is actually lower than my old takeout spending.",
        "The author recently subscribed to a weekly produce delivery service.",
    ),
    (
        "The local farmer's market reopens every Saturday from May through October and is a favourite weekend stop for our family.",
        "The town also hosts a small jazz festival every August.",
    ),
    (
        "Reading thirty minutes before bed instead of scrolling on my phone has dramatically improved both my sleep quality and my mood.",
        "The author keeps a stack of unread books on the bedside table.",
    ),
    (
        "I built a small treehouse in the backyard last summer using lumber salvaged from a neighbour's barn renovation.",
        "The backyard also features a vegetable garden and a compost bin.",
    ),
    (
        "Our weekly board game night with the neighbours has become the social highlight of the week for everyone involved.",
        "One of the neighbours teaches piano in their home studio.",
    ),
]


# Map doc_types to (entail, contradict, neutral) triples.
#
# DocTypeLiteral has six values: spec, runbook, rfc, legal, academic, narrative.
# Educational science prose maps onto "rfc" (informational standards-style
# writing) and blog/tech how-to prose maps onto "runbook" (operational,
# step-and-rationale prose). This is consistent with how the eval substrate
# already uses these labels across cross_doc_coverage and merge cases.
DOC_TYPE_CASES: dict[
    str, tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]
] = {
    "spec": (SPEC_ENTAIL, SPEC_CONTRADICT, SPEC_NEUTRAL),
    "legal": (LEGAL_ENTAIL, LEGAL_CONTRADICT, LEGAL_NEUTRAL),
    "academic": (ACADEMIC_ENTAIL, ACADEMIC_CONTRADICT, ACADEMIC_NEUTRAL),
    "rfc": (EDU_ENTAIL, EDU_CONTRADICT, EDU_NEUTRAL),
    "narrative": (NARR_ENTAIL, NARR_CONTRADICT, NARR_NEUTRAL),
    "runbook": (TECH_ENTAIL, TECH_CONTRADICT, TECH_NEUTRAL),
}


def build_dataset() -> list[dict[str, str]]:
    """Assemble 200 cases balanced across labels and doc types.

    6 doc types x (11 entail + 11 contradict + 11 neutral) = 198 base
    cases. We add an extra entailment (spec) and an extra neutral
    (runbook) case to land at exactly 200 with the per-class balance
    inside the test's tolerance window.
    """
    out: list[dict[str, str]] = []
    counter = 1
    for doc_type, (ent, con, neu) in DOC_TYPE_CASES.items():
        assert len(ent) >= 11 and len(con) >= 11 and len(neu) >= 11, doc_type
        # Take exactly 11 of each per doc type (66 per class, 198 total).
        for premise, hypothesis in ent[:11]:
            out.append(_case(counter, premise, hypothesis, "entailment", doc_type))
            counter += 1
        for premise, hypothesis in con[:11]:
            out.append(_case(counter, premise, hypothesis, "contradiction", doc_type))
            counter += 1
        for premise, hypothesis in neu[:11]:
            out.append(_case(counter, premise, hypothesis, "neutral", doc_type))
            counter += 1

    # Top up to 200 with one extra entail (spec) and one extra
    # neutral (runbook) — keeps the per-class balance within ±2 of 66.
    out.append(_case(counter, SPEC_ENTAIL[11][0], SPEC_ENTAIL[11][1], "entailment", "spec"))
    counter += 1
    out.append(_case(counter, TECH_NEUTRAL[11][0], TECH_NEUTRAL[11][1], "neutral", "runbook"))

    return out


def _case(idx: int, premise: str, hypothesis: str, label: str, doc_type: str) -> dict[str, str]:
    return {
        "id": f"cal-{idx:03d}",
        "premise": premise,
        "hypothesis": hypothesis,
        "gold_label": label,
        "doc_type": doc_type,
    }


def main() -> None:
    cases = build_dataset()
    out_path = (
        Path(__file__).resolve().parent.parent.parent / "tests" / "eval" / "calibration_eval.jsonl"
    )
    with out_path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} cases to {out_path}")

    # Sanity-print label balance.
    labels = [c["gold_label"] for c in cases]
    for label in ("entailment", "contradiction", "neutral"):
        print(f"  {label}: {labels.count(label)}")


if __name__ == "__main__":
    main()
