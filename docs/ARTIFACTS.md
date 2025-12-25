# 🎭 The Artifact Gallery

## What Are Artifacts?
When AI gets... creative. Here's what to expect:

### 1. 👥 The Clone Saga
**Symptom**: Two (or more) heads, merged faces
**Cause**: High resolution (1024px+)
**Fix**: Use 512x768 + upscale
**Example**: [artifact_two_heads.png]

### 2. 🖐️ Hand Horror
**Symptom**: 6+ fingers, backwards thumbs, hand-feet
**Cause**: SD was trained on diverse hand poses (confusing!)
**Fix**: Add "normal hands, five fingers" to prompt
**Pro Tip**: Use ControlNet Openpose (coming soon!)

### 3. 📝 Almost-Words
**Symptom**: Text looks like "ČÅFĒ" instead of "CAFE"
**Cause**: SD wasn't trained for text rendering
**Fix**: Add text in post-processing, or use SDXL

### 4. 🦵 Anatomical Mysteries
**Symptom**: Extra limbs, twisted joints
**Cause**: Multiple people in training data merged
**Fix**: Higher CFG scale (12-15), better negative prompts

### 5. 🌈 The "Cursed" Generation
**Symptom**: Completely abstract chaos
**Cause**: Conflicting prompt terms
**Example**: "realistic anime pixel art photograph"
**Fix**: Pick one style!

## Prevention Tips
1. **Start small**: 512x512 → upscale
2. **Strong negatives**: Use anti-artifact preset
3. **Simple prompts**: "anime girl" > "anime girl with 17 accessories"
4. **Test seeds**: Bad seed? Try another!
5. **CFG sweet spot**: 8-12 (not 20!)

## When Artifacts Are Good
Sometimes artifacts are hilarious/artistic:
- Abstract art mode
- Surrealist experiments
- Meme material 😂
