curl http://localhost:26500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi3samba",
    "messages": [
      {"role": "user", "content": "Please reason step by step, and put your final choice of one letter from A/B/C/D as in Answer: $LETTER. Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so thatthey be clearly resolved?\n\nA) 10^-4 ev\nB) 10^-8 ev\nC) 10^-9 ev\nD) 10^-11 ev"}
    ],
    "max_tokens": 32768,
    "temperature": 0.6,
    "top_p": 0.95
  }'
# "how many r'\''s in \"strawberrry\"?"

# Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nIn a parallel universe where a magnet can have an isolated North or South pole, Maxwell’s equations look different. But, specifically, which of those equations are different?\n\nA) The ones related to the circulation of the electric field and the divergence of the magnetic field.\nB) The ones related to the divergence and the curl of the magnetic field.\nC) The one related to the divergence of the magnetic field.\nD) The one related to the circulation of the magnetic field and the flux of the electric field.
