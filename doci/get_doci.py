def contract_2e(eri, civec, norb, nelec, link_index=None):
    '''Compute E_{pq}E_{rs}|CI>'''
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    assert(neleca == nelecb)

    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = link_index.shape[0]
    t1 = np.zeros((norb,na))
    t2 = np.zeros((norb,norb,na))
    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            t1[i,str1] += civec[str0]
    #:        else:
    #:            t2[a,i,str1] += civec[str0]
    link1 = link_index[link_index[:,:,0] == link_index[:,:,1]].reshape(na,-1,4)
    link2 = link_index[link_index[:,:,0] != link_index[:,:,1]].reshape(na,-1,4)
    t1[link1[:,:,1],link1[:,:,2]] = civec[:,None]
    t2[link2[:,:,0],link2[:,:,1],link2[:,:,2]] = civec[:,None]

    eri = ao2mo.restore(1, eri, norb)
    # [(ia,ja|ib,jb) + (ib,jb|ia,ja)] ~ E_{ij}E_{ij} where i != j
    cinew = np.einsum('ijij,ijp->p', eri, t2) * 2

    # [(ia,ja|ja,ia) + (ib,jb|jb,ib)] ~ E_{ij}E_{ji} where i != j
    k_diag = np.einsum('ijji->ij', eri)
    t2 = np.einsum('ij,ijp->ijp', k_diag * 2, t2)

    # [(ia,ia|ja,ja) + (ia,ia|jb,jb) + ...] ~ E_{ii}E_{jj}
    j_diag = np.einsum('iijj->ij', eri)
    t1 = np.einsum('ij,jp->ip', j_diag, t1) * 4

    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            cinew[str0] += t1[i,str1]
    #:        else:
    #:            cinew[str0] += t2[a,i,str1]
    cinew += np.einsum('pi->p', t1[link1[:,:,1],link1[:,:,2]])
    cinew += np.einsum('pi->p', t2[link2[:,:,0],link2[:,:,1],link2[:,:,2]])
    return cinew

def make_hdiag(h1e, eri, norb, nelec, opt=None):
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    occslista = cistring._gen_occslst(range(norb), neleca)
    eri = ao2mo.restore(1, eri, norb)
    diagj = np.einsum('iijj->ij',eri)
    diagk = np.einsum('ijji->ij',eri)
    hdiag = []
    for aocc in occslista:
        bocc = aocc
        e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
        e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
           + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
           - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
        hdiag.append(e1 + e2*.5)
    return np.array(hdiag)

def kernel(h1e, eri, norb, nelec, ecore=0):
    return DOCI().kernel(h1e, eri, norb, nelec, ecore=ecore)

def make_rdm1(civec, norb, nelec, link_index=None):
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    assert(neleca == nelecb)

    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = cistring.num_strings(norb, neleca)
    t1 = np.zeros((norb,na))
    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            t1[i,str1] += civec[str0]
    link1 = link_index[link_index[:,:,0] == link_index[:,:,1]].reshape(na,-1,4)
    t1[link1[:,:,1],link1[:,:,2]] = civec[:,None]

    dm1 = np.diag(np.einsum('ip,p->i', t1, civec)) * 2
    return dm1

def make_rdm12(civec, norb, nelec, link_index=None, reorder=True):
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    assert(neleca == nelecb)

    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = cistring.num_strings(norb, neleca)
    t1 = np.zeros((norb,na))
    t2 = np.zeros((norb,norb,na))
    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            t1[i,str1] += civec[str0]
    #:        else:
    #:            t2[a,i,str1] += civec[str0]
    link1 = link_index[link_index[:,:,0] == link_index[:,:,1]].reshape(na,-1,4)
    link2 = link_index[link_index[:,:,0] != link_index[:,:,1]].reshape(na,-1,4)
    t1[link1[:,:,1],link1[:,:,2]] = civec[:,None]
    t2[link2[:,:,0],link2[:,:,1],link2[:,:,2]] = civec[:,None]

    idx = np.arange(norb)
    dm2 = np.zeros([norb]*4)
    # Assign to dm2[i,j,i,j]
    dm2[idx[:,None],idx,idx[:,None],idx] += 2 * np.einsum('ijp,p->ij', t2, civec)
    # Assign to dm2[i,j,j,i]
    dm2[idx[:,None],idx,idx,idx[:,None]] += 2 * np.einsum('ijp,ijp->ij', t2, t2)
    # Assign to dm2[i,i,j,j]
    dm2[idx[:,None],idx[:,None],idx,idx] += 4 * np.einsum('ip,jp->ij', t1, t1)

    dm1 = np.einsum('ijkk->ij', dm2) / (neleca+nelecb)

    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2


