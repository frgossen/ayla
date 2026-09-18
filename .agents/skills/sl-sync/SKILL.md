---
name: sl-sync
description: Clean and synchronize Sapling state in the frgossen/ayla repository after PR work by refreshing main, removing duplicate or completed PR commits from the smartlog, and rebasing active local stacks. Use when the user asks to clean or sync the local sl tree; do not use for general Git operations or other repositories.
---

# Sapling sync

Leave the Ayla checkout clean and based on the latest remote `main`, with only
active local work visible in the Sapling smartlog.

## Scope and safety

- Verify that `sl root` succeeds and that `sl paths` identifies
  `github.com/frgossen/ayla`. Stop without changing anything in any other
  repository.
- Treat a request to run this workflow as authorization for local Sapling
  cleanup only. Do not push, force-push, create, close, or otherwise modify
  pull requests unless the user separately requests it.
- Inspect `sl status` first. Do not rebase, update, or hide anything while the
  working copy has uncommitted changes; report them and ask how to preserve
  them.
- Never rewrite public commits. Before hiding a commit, confirm that it is not
  the working-copy parent and is not required as an ancestor of retained local
  work. Remember that `sl hide` also hides descendants and deletes bookmarks
  pointing to hidden commits.
- Treat commit hashes as recovery handles. Include every explicitly hidden
  commit in the final summary; hidden commits can be recovered with
  `sl unhide`.

## Workflow

1. Inspect the current smartlog, visible draft commits, bookmarks, obsolete
   markers, and any interrupted operation. Establish which local stacks are
   active before mutating state.
2. Pull the `main` bookmark from the default remote and identify the refreshed
   remote-main commit. Do not infer freshness from a stale local bookmark.
3. Associate visible draft commits with pull requests using commit hashes,
   bookmarks or branch names, and read-only GitHub queries. Distinguish:
   - active commits belonging to open PRs or unpublished work;
   - older rewritten versions superseded by a visible successor;
   - commits belonging to merged PRs whose result is present in refreshed
     `main`;
   - commits belonging to closed PRs that are no longer the checkout or an
     ancestor required by active work.
4. Rebase every retained independent local stack onto refreshed `main`, using
   the stack root as the source so descendant order is preserved. Preview the
   selected revisions before running `sl rebase`. Let Sapling rebase out
   changes already present in the destination.
5. Resolve straightforward conflicts only when the intended result is clear.
   For ambiguous conflicts, stop and report them without guessing. Do not
   abort an interrupted operation that predates this run.
6. Run `sl hide --cleanup` to remove obsolete rewritten versions. Explicitly
   hide any remaining completed-PR commits only after checking their draft
   descendants and confirming that retained work no longer depends on them.
7. Reinspect `sl status`, visible draft commits, bookmarks, and the bare `sl`
   smartlog. The final state should have an empty working-copy status, fresh
   `main`, each active PR or unpublished stack represented once, and no stale
   merged, closed, or superseded PR commits.

Report the refreshed `main` commit, rebased stacks, hidden commit hashes and
their PR disposition, plus anything intentionally retained or requiring user
attention.
