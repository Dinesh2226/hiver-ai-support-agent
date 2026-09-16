# Failure Analysis

Analysis based on the 200-example golden intent benchmark and 30-example reply-quality judge sample.

## Intent Classification

- Golden examples: **200**
- Incorrect intent predictions: **88**

### Largest confusion pairs

| Gold | Predicted | Count |
|---|---|---:|
| `ORDER_DELIVERY` | `COMPLAINT_FEEDBACK` | 22 |
| `COMPLAINT_FEEDBACK` | `OTHER` | 8 |
| `ORDER_DELIVERY` | `OTHER` | 7 |
| `ORDER_DELIVERY` | `PRIME_MEMBERSHIP` | 4 |
| `PRODUCT_CONTENT` | `OTHER` | 4 |
| `RETURN_REFUND` | `COMPLAINT_FEEDBACK` | 4 |
| `PAYMENT_BILLING` | `COMPLAINT_FEEDBACK` | 4 |
| `PRODUCT_CONTENT` | `COMPLAINT_FEEDBACK` | 4 |
| `OTHER` | `COMPLAINT_FEEDBACK` | 3 |
| `ACCOUNT_ACCESS` | `OTHER` | 3 |

### Representative classification errors

- **Row 1**: gold=`ORDER_DELIVERY`, predicted=`COMPLAINT_FEEDBACK`
  - Message: Also @AmazonHelp - @134825 ist schon a bissl nervig vong zustellservice - null benachrichtigung und jetzt kann ich keinens shop auswählen 😞
- **Row 2**: gold=`PAYMENT_BILLING`, predicted=`OTHER`
  - Message: Amazonは流石に分割払い出来ない、よな？？
- **Row 6**: gold=`PAYMENT_BILLING`, predicted=`RETURN_REFUND`
  - Message: @AmazonHelp Juste marre de jouer au chat et souris .... Pour moi c'est pas grave....Des offres y'en a partout mais au vu de la manière...Les 3 ou 4 retraits "incompréhensibles" d'amazon, on demandera le remboursement Justificatifs à l'appui...On l...
- **Row 8**: gold=`COMPLAINT_FEEDBACK`, predicted=`OTHER`
  - Message: @AmazonHelp Guess you guys didn’t have a reply to that info eh lol
- **Row 9**: gold=`ORDER_DELIVERY`, predicted=`COMPLAINT_FEEDBACK`
  - Message: @AmazonHelp Here is my reply.. Thank you for ruining my Diwali by claiming false commitments.. It was my mistake to order something from you.. https://t.co/W5vU6xy0N1
- **Row 13**: gold=`ORDER_DELIVERY`, predicted=`COMPLAINT_FEEDBACK`
  - Message: @AmazonHelp @360907 Your support team just escalates the issue and actually nothing much happens after that...pathetic delivery services.
- **Row 17**: gold=`RETURN_REFUND`, predicted=`COMPLAINT_FEEDBACK`
  - Message: DISSATISFIED, Ordered 2 product, received 1, - declined the order. Told to make a new Order- then the offer on product was ended- called customer care- no good response- Told I should have kept the 1 Order by paying for 2 https://t.co/dOZJ1L1IlU @...
- **Row 19**: gold=`PAYMENT_BILLING`, predicted=`COMPLAINT_FEEDBACK`
  - Message: @AmazonHelp Day 16 Rajni from the cs team promised yesterday if the pay balance is not reflecting in 24 hours she will add it manually immediately, guess what! Today she says she can't and I'll have to wait for 24 hours!! What a joke!! Amazon look...
- **Row 21**: gold=`OTHER`, predicted=`COMPLAINT_FEEDBACK`
  - Message: @AmazonHelp Nope... 😔 It was a gift....
- **Row 22**: gold=`ORDER_DELIVERY`, predicted=`COMPLAINT_FEEDBACK`
  - Message: @115821 why am I paying for prime when you can't get my packages here on time? #GiftsForAFamilyinNeed
- **Row 23**: gold=`RETURN_REFUND`, predicted=`COMPLAINT_FEEDBACK`
  - Message: I'm really getting tired of @115821 sending me used items when I pay for new. This has happened at least 4 times out of the last 6 orders.
- **Row 26**: gold=`ORDER_DELIVERY`, predicted=`PRIME_MEMBERSHIP`
  - Message: @115821 As a prime member it is unacceptable that items promised to be delivered within two business days are not arriving on time.

## Top 5 Failure Modes

### 1. Operational delivery issue vs complaint language
**Root cause:** Customers often combine a concrete delivery problem with strong frustration or criticism. The model can overweight complaint-style wording.
**Mitigation:** Prioritize the underlying operational problem over tone and add counterexamples containing strong emotional language.

### 2. Complaint/feedback vs OTHER
**Root cause:** Short conversational messages can lack enough context to distinguish dissatisfaction from general commentary.
**Mitigation:** Add boundary examples and use clarification-first behavior when intent evidence is weak.

### 3. Rare-intent undercoverage
**Root cause:** Rare intents have limited training and evaluation support, making their decision boundaries less stable.
**Mitigation:** Collect more targeted examples for rare intents and use class-aware sampling.

### 4. Multilingual and context-poor messages
**Root cause:** Short Japanese, French, German, and other multilingual messages provide weaker lexical evidence for the current TF-IDF/retrieval pipeline.
**Mitigation:** Add multilingual examples and evaluate language-specific performance.

### 5. Reply generation grounding failures
**Root cause:** Generated replies can become generic or introduce facts that are not supported by the customer message or evidence.
**Mitigation:** Use evidence-first prompting, prohibit unsupported claims, and ask a targeted clarification question when evidence is insufficient.

## Reply Generation Evaluation

- Replies judged: **30**
- Mean overall score: **1.87/5**
- Critical errors: **21/30**

### Lowest-scoring reply examples

- **Row 20** — score=1.0/5
  - The reply is generic and does not address the customer's specific message about purchasing a Hagaren art book on Amazon. It lacks any meaningful engagement with the customer's intent or context, and there is no evidence to support the re...
- **Row 33** — score=1.0/5
  - The reply makes unsupported claims about the availability of the Amazon Fire TV Stick in Canada and provides a link to a third-party website, which is not supported by the evidence. It also fails to address the customer's specific concer...
- **Row 25** — score=1.0/5
  - The reply does not address the core issue of account access, fails to use the evidence provided, and does not ask for the necessary clarification about the error message or email confirmation.
- **Row 22** — score=1.0/5
  - The reply makes unsupported claims about delivery dates and processing times without any evidence from the provided historical data. It invents details about the customer's account or delivery status, which is not allowed. The reply fail...
- **Row 34** — score=1.0/5
  - The reply is not grounded in the customer's message or the evidence. The customer explicitly states they have already received everything, but the reply assumes they need more assistance without clarification. The reply lacks helpfulness...
- **Row 96** — score=1.0/5
  - The reply is not grounded in the customer's message or the provided evidence. It fails to address the specific complaints about the return/refund process, fraud sellers, or the frustration with service. It also does not use the evidence ...
- **Row 105** — score=1.0/5
  - The reply is not grounded in the evidence or the customer's message. The customer has already contacted customer support, sent emails, and called, and the reply does not address the specific issue of being charged after cancellation or t...
- **Row 109** — score=1.0/5
  - The reply fails to address the customer's specific concerns about the delivery status, the confusion with the 'signed' mark, and the potential misplacement of packages. It does not use the available evidence or clarify the customer's iss...
- **Row 89** — score=1.0/5
  - The reply lacks any direct acknowledgment of the customer's issue or attempt to resolve it. It is generic and does not use the provided evidence to address the problem. It also fails to ask for clarification or offer a next step.
- **Row 167** — score=1.0/5
  - The reply is not grounded in the customer's message or evidence. It makes no attempt to address the specific complaint about fraud, missing product, or payment issues. It fails to ask for clarification or take any action based on the pro...

## What is misleading about my headline number?

Intent accuracy alone does not represent end-to-end support quality. The 200-example golden set is relatively small and imbalanced, while several rare intents have very low support.

The reply-generation score is also diagnostic rather than ground truth because the judge is itself an automated LLM. Human-reviewed ratings should therefore be used to validate the judge rather than treating its absolute score as a definitive customer-quality measurement.

## Next-Week Plan

1. Add targeted boundary examples for the highest-confusion intent pairs.
2. Expand rare-intent and multilingual annotation.
3. Add deterministic precedence rules for high-signal operational issues.
4. Improve multilingual retrieval once a supported runtime is available.
5. Add an evidence/unsupported-claim validator after generation.
6. Expand human reply evaluation and periodically re-check LLM-judge agreement.