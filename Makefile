enable-git-hooks:
	git config core.hooksPath .githooks
	$(info Git hooks in .githooks enabled, you should really review the hooks if you haven't already)
