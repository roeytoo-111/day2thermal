Combined GAN pair set (hard links; built 2026-09-30).
train = 2026-07-08 (all 392 pairs, vid_*) + 2026-06-23 (all 321, s0623_*)
val   = 2026-07-15 (all 292, s0715_*) -- a whole unseen recording day: tests generalisation, not memorisation.
Leak rule: a translator trained here must not produce detector training data from 07-08 RGB (07-08 is the
detector test video) nor be evaluated on 07-15 as if unseen after retraining on it.
