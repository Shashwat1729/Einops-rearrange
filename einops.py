from typing import Dict, List, Tuple, Optional
import numpy as np

class ee(Exception): # Einops error
    pass

class PatternParser: # Pattern parser
    def __init__(self, p: str):
        self.p = p.strip()
        self.val_pat()
        self.hel = '...' in p
    
    def parse(self) -> Tuple[List[str], List[str]]: # Split in and out axes
        if '->' not in self.p:
            raise ee("Arrow not found")
        
        pts = self.p.split('->')
        ip = pts[0].strip()
        op = pts[1].strip()
        
        return self.par_axis(ip), self.par_axis(op)
    
    def par_axis(self, p: str) -> List[str]: # Split axes
        if '...' not in p:
            return self.pap(p)
            
        # Now we split on ellipsis and parse parts
        pts = [x.strip() for x in p.split('...', 1)]
        a = []
        for pt in pts:
            if pt:
                a.extend(self.pap(pt))
            
        # Now we place ellipsis in right spot
        if p.startswith('...'):
            return ['...'] + a
        else:
            return a + ['...']
    
    def pap(self, p: str) -> List[str]: # Parse pattern part
        pts = []
        i = 0
        j = 0
        
        for k, l in enumerate(p):
            if l == '(':
                j += 1
            elif l == ')':
                j -= 1
            elif l.isspace() and j == 0:
                if i < k:
                    pts.append(p[i:k].strip())
                i = k + 1
                
        if i < len(p):
            pts.append(p[i:].strip())
            
        return [x for x in pts if x]
    
    def val_pat(self) -> None: # Check pattern valid
        if not self.p or '->' not in self.p:
            raise ee("Pattern not valid")
            
        # Now we check if string has invalid characters or not
        if not all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789() .->' for c in self.p):
            raise ee("Invalid characters")
            
        # Now we check if brackets match
        if self.p.count('(') != self.p.count(')'):
            raise ee("Brackets not matching")

class OperationParser: # Op parser
     
    def par_split(a: str) -> Tuple[str, List[str]]: # Split op
        # Now we check if there's a bracket
        if not a.startswith('('):
            return a, []
        
        # Now we get inner parts
        inn = a[1:-1].strip()
        sa = inn.split()
        return a, sa

    def par_merge(a: List[str]) -> Tuple[List[str], str]: # Merge op
        if not a:
            raise ee("Empty merge") 
        else:
            return a, ''.join(a)
     
    def par_repeat(a: str) -> Tuple[str, str, int]: # Repeat op
        if a.isdigit():
            return '', a, int(a)
        else:
            return a, '', 0

def validate_shape(t: np.ndarray, ia: List[str], **al) -> None: # Check shapes
    s = t.shape
    # Now we deal with ellipsis
    na = [x for x in ia if x != '...']
    if '...' in ia:
        if len(s) < len(na):
            raise ee("Need more dimensions")
    elif len(s) != len(ia):
        raise ee("Mismatch in dimensions")

    # Now we check each dimension
    i = 0
    for x in ia:
        if x == '...':
            i = len(s) - (len(ia) - ia.index('...') - 1)
            continue
        
        if x.startswith('('):
            pts = x[1:-1].strip().split()
            sz = 1
            u = []
            
            for p in pts:
                if p in al:
                    sz *= al[p]
                else:
                    u.append(p)
            
            if len(u) > 1:
                raise ee("Unknowns not matching")
            if len(u) == 1 and s[i] % sz != 0:
                raise ee("Size mismatch")
            if not u and s[i] != sz:
                raise ee("Size mismatch")
        
        elif x in al and s[i] != al[x]:
            raise ee("Wrong axis size")
            
        i += 1

def apply_operations(t: np.ndarray, ia: List[str], oa: List[str], **al) -> np.ndarray:
    # Now we transform tensor shape
    r = t.copy()
    ca = ia.copy()
    
    # Now we handle ellipsis
    if '...' in ca:
        ei = ca.index('...')
        bd = len(t.shape) - (len(ca) - 1)
        ba = [f'b{i}' for i in range(bd)]
        ca[ei:ei+1] = ba
        
        # Now we update batch dimensions
        if '...' in oa:
            oei = oa.index('...')
            oa[oei:oei+1] = ba
    
    # Now we do splits
    i = 0
    while i < len(ca):
        a = ca[i]
        if a.startswith('('):
            og, sa = OperationParser.par_split(a)
            if sa:
                # Now we get sizes
                ts = r.shape[i]
                ss = []
                ks = 1
                ua = None
                
                # Now we collect known sizes
                for x in sa:
                    if x in al:
                        sz = al[x]
                        ss.append(sz)
                        ks *= sz
                    else:
                        ua = x
                        ss.append(None)
                
                # Now we compute unknown size
                if ua is not None:
                    us = ts // ks
                    for j, sz in enumerate(ss):
                        if sz is None:
                            ss[j] = us
                
                # Now we do reshape
                ns = list(r.shape)
                ns[i:i+1] = ss
                r = r.reshape(ns)
                ca[i:i+1] = sa
        i += 1
    
    # Now we do merges
    i = 0
    while i < len(oa):
        if oa[i].startswith('('):
            og, ma = OperationParser.par_split(oa[i])
            if ma:
                # Now we find merge pos
                mp = []
                for x in ma:
                    try:
                        p = ca.index(x)
                        mp.append(p)
                    except ValueError:
                        raise ee("Axis not found")
                
                # Check if axes to merge are contiguous in current order
                mp.sort()
                if mp[-1] - mp[0] + 1 != len(mp):
                    # Now we make adjacent
                    for j, p in enumerate(mp):
                        if p != mp[0] + j:
                            mo = list(range(len(ca)))
                            mo.pop(p)
                            mo.insert(mp[0] + j, p)
                            r = np.transpose(r, mo)
                            ca = [ca[k] for k in mo]
                
                # Now we do merge
                ns = list(r.shape)
                ms = np.prod([ns[mp[0] + j] for j in range(len(mp))])
                ns[mp[0]:mp[-1] + 1] = [ms]
                r = r.reshape(ns)
                ca[mp[0]:mp[-1] + 1] = [oa[i]]
        i += 1
    
    # Now we do repeats
    i = 0
    while i < len(ca):
        a = ca[i]
        if a == '1':
            # Now we get repeat size
            if i < len(oa):
                na = oa[i]
                if na in al:
                    rc = al[na]
                    ns = list(r.shape)
                    ns[i] = rc
                    r = np.repeat(r, rc, axis=i)
                    ca[i] = na
        i += 1
    
    # Now we do final transpose
    if ca != oa:
        # Now we map axes
        ap = {}
        for i, a in enumerate(oa):
            if not a.startswith('('):
                ap[a] = i
        
        # Now we add batch
        for i, a in enumerate(ca):
            if a.startswith('b'):
                if a not in ap:
                    ap[a] = i
        
        # Now we do transpose
        try:
            no = [ap[a] for a in ca]
            r = np.transpose(r, no)
            
        except KeyError as e:
            raise ee("Axis not found")
    
    return r

def rearrange(t: np.ndarray, p: str, **al) -> np.ndarray:
    # Now we rearrange tensor
    ia, oa = PatternParser(p).parse()
    validate_shape(t, ia, **al)
    return apply_operations(t, ia, oa, **al)
