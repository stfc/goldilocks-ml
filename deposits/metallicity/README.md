# Metallicity models

Networks trained on whether a material conducts, filed by what each artifact
gives you.

- `is_metal/` — models that answer the question: metal or insulator.
- `representation/` — models published for the learned vector they produce,
  which another model consumes as input features.

Both directories hold the same architecture trained on the same kind of label.
They are separated because a consumer uses them completely differently: one
returns a decision, the other returns 64 numbers that mean nothing on their own.
