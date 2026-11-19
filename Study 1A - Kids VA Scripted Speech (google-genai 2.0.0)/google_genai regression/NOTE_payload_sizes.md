# Reading the captured payload sizes

Three payload figures appear across these files and they measure different things.

- `gemini_out_v0_4_0.jsonl`, meta line. A single recording sent under
  google-genai 0.4.0. The wire payload was 472,080 characters. Decoding it once
  yields 354,060 characters that are not FLAC. Decoding a second time yields
  FLAC. The 354,060-character string is exactly the payload a correct single
  encoding produces, which is why the paper reports 472,080 against 354,060 for
  this recording. The ratio is exactly four thirds.
- `study1_gemini_wire_forensics.json`. A capture from a separate 252-call run
  under google-genai 2.0.0. Its payload length of 369,768 characters belongs to
  a different recording and should not be compared against the numbers above.
