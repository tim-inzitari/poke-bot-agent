# Alakazam recent-20 RTP overlay v1 handoff

Status: corpus preparation complete; no checkpoint was loaded and no RTP
training, MCTS, simulator search, service change, selector change, or transfer
to Inzi was performed.

## Sealed inputs and outputs

- Source window: 2026-07-23 through 2026-08-11 inclusive.
- Base corpus manifest: `sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1`.
- Base pack completion: `sha256:e9756ba8fbf6f813778c4ce03af44b22b653e00586bfdb0c917a7313380ce5ba`.
- Base schema: `sha256:3a528138e819b10691e8a7ed917c55e4000b9ec039562cf859cc2e00706bb3fa`.
- Overlay schema: `sha256:29de1530768f1b3f8b9be7e02fe2dfef3eeb64475d1ba0ff146026d5c54d6a37`.
- Overlay manifest: `sha256:081e40d9b9cc98714abaa8945c8d176a9143bdb8e87aeeee0327878642b118bd`.
- Completion receipt: `sha256:c7a9392a1c91adfa27730963d867ee88069c41585d3fd2027df96d2301edfd91`.
- Existing-pipeline validation: `sha256:4b1611013154f27f4be7f097ba2cd692504f00a2c91c65313e8d3a2cb2bf069b`.

The artifact root on Elmo is
`/srv/poke-bot-agent/outputs/experiments/alakazam-recent20-rtp-overlay-v1-attempt4`.
The manifest is the entry point. It references, but does not copy, the sealed
40-wide base feature tensors.

## Remaining sidecar work

1. Select an immutable frozen parent checkpoint and record its full path,
   SHA-256, architecture/config identity, and selection receipt. Do not bind a
   mutable selector or active-training checkpoint.
2. Receipt-bind the established RTP encoder/projection to the base pack's
   40-wide option features. Keep the overlay adapter and public-information
   masks unchanged; any new projection must be versioned separately.
3. Instantiate `ArchetypeRTPJob` with the manifest digest, base-pack root, base
   completion digest, and the selected frozen checkpoint. Preserve the sealed
   whole-day train/validation/evaluation split.
4. Run an explicitly authorized CPU/GPU sizing smoke, then train only the RTP
   sidecar. Do not change the current policy, selectors, r274, or production
   authority.
5. Validate held-out evaluation behavior, deterministic resume, public-only
   inputs, masked unchosen actions, resource use, and checkpoint/manifest
   binding. Publish content-addressed model and validation receipts before any
   promotion proposal.

