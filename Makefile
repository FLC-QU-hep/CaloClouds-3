enable-git-hooks:
	find .githooks -type f -exec chmod +x {} \;
	git config core.hooksPath .githooks
	$(info Git hooks in .githooks enabled, you should really review the hooks if you haven't already)
