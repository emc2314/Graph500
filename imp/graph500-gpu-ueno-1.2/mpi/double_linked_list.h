/*
 * Copyright (C) Koji Ueno 2012-2013.
 *
 * This file is part of Graph500 Ueno implementation.
 *
 *  Graph500 Ueno implementation is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Graph500 Ueno implementation is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Graph500 Ueno implementation.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef DOUBLE_LINKED_LIST_H_
#define DOUBLE_LINKED_LIST_H_

#include <stddef.h>

template<typename Type>
void inline initializeListHead(Type* head)
{
	head->fLink = head->bLink = head;
}

template<typename Type>
void inline initializeListEntry(Type* entry)
{
#ifndef NDEBUG
	entry->bLink = NULL;
	entry->fLink = NULL;
#endif
}

template<typename Type>
bool inline listIsEmpty(Type* head)
{
	return head->bLink == head;
}

template<typename Type>
void inline listRemove(Type* entry)
{
	Type* fLink = entry->fLink;
	Type* bLink = entry->bLink;
	entry->bLink->fLink = fLink;
	entry->fLink->bLink = bLink;
#ifndef NDEBUG
	entry->bLink = NULL;
	entry->fLink = NULL;
#endif
}

template<typename Type>
void inline listInsertFoward(Type* pos, Type* newentry)
{
#ifndef NDEBUG
	assert (newentry->bLink == NULL);
	assert (newentry->fLink == NULL);
#endif
	Type* fLink = pos->fLink;
	Type* bLink = pos;
	newentry->fLink = fLink;
	newentry->bLink = bLink;
	fLink->bLink = newentry;
	bLink->fLink = newentry;
}

template<typename Type>
void inline listInsertBack(Type* pos, Type* newentry)
{
#ifndef NDEBUG
	assert (newentry->bLink == NULL);
	assert (newentry->fLink == NULL);
#endif
	Type* fLink = pos;
	Type* bLink = pos->bLink;
	newentry->fLink = fLink;
	newentry->bLink = bLink;
	fLink->bLink = newentry;
	bLink->fLink = newentry;
}

struct ListEntry {
	ListEntry *fLink, *bLink;
};

#define CONTAINING_RECORD(address, type, field) ((type*)((char*)(address)-offsetof(type,field)))

#endif /* DOUBLE_LINKED_LIST_H_ */
