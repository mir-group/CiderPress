import numpy as np


class CiderParallelization:
    def __init__(self, comm, nalpha):
        self._comm = self.world = comm  # typically world
        self.comm = None  # might be shaped differently
        self.size = comm.size
        self.nalpha = nalpha

        self.alpha = None
        self.domain = None

        self.ps_comm = None
        self.pa_comm = None

        self.nclaimed = 1
        self.navail = comm.size

    def set(self, alpha=None, domain=None):
        if alpha is not None:
            self.alpha = alpha
        if domain is not None:
            self.domain = domain

        nclaimed = 1
        for group, name in zip([self.alpha, self.domain], ["alpha", "domain"]):
            if group is not None:
                assert (
                    group > 0
                ), "Bad: Only {} cores requested for " "{} parallelization".format(
                    group, name
                )
                # if self.size % group != 0:
                #    msg =  ('Cannot parallelize as the '
                #            'communicator size %d is not divisible by the '
                #            'requested number %d of ranks for %s '
                #            'parallelization' % (self.size, group, name))
                #    raise ValueError(msg)
                nclaimed *= group
        navail = self.size // nclaimed

        # assert self.size % nclaimed == 0
        # assert self.size % navail == 0

        self.navail = navail
        self.nclaimed = nclaimed

    def get_communicator_sizes(self, alpha=None, domain=None):
        self.set(alpha=alpha, domain=domain)
        self.autofinalize()
        return self.set(alpha=alpha, domain=domain)

    def build_communicators(self, alpha=None, domain=None, order="ad"):
        self.set(alpha=alpha, domain=domain)
        self.autofinalize()

        comm = self.comm
        if comm is None:
            self.comms = {"a": None, "d": None}
            return self.comms
        rank = comm.rank
        communicators = {}
        parent_stride = self.size
        offset = 0

        groups = dict(a=self.alpha, d=self.domain)

        for name in order:
            group = groups[name]
            stride = parent_stride // group
            # First rank in this group
            r0 = rank % stride + offset
            # Last rank in this group
            r1 = r0 + stride * group
            ranks = np.arange(r0, r1, stride)
            communicators[name] = comm.new_communicator(ranks)
            parent_stride = stride
            # Offset for the next communicator
            offset += communicators[name].rank * stride

        self.comms = communicators

        self.alpha_loc_list = []
        asize = self.comms["a"].size
        for r in range(asize):
            self.alpha_loc_list.append(int(np.ceil((float(r) * self.nalpha) / asize)))
        self.alpha_loc_list.append(self.nalpha)

        return communicators

    def autofinalize(self):
        if self.alpha is None:
            self.set(alpha=self.get_optimal_alpha_parallelization())
        if self.domain is None:
            self.set(domain=self.navail)

        self.comm = self._comm.new_communicator(np.arange(self.alpha * self.domain))

    def get_optimal_alpha_parallelization(self, alpha_prio=1.4):
        if self.domain:
            ncpus = min(self.nalpha, self.navail)
            return ncpus
        else:
            ncpus = min(self.nalpha, self.navail)
            return ncpus
        ncpuvalues, wastevalues = self.find_alpha_parallelizations()
        scores = ((self.navail // ncpuvalues) * ncpuvalues**alpha_prio) ** (
            1.0 - wastevalues
        )
        arg = np.argmax(scores)
        ncpus = ncpuvalues[arg]
        return ncpus

    def find_alpha_parallelizations(self):
        nalpha = self.nalpha
        ncpuvalues = []
        wastevalues = []

        ncpus = nalpha
        while ncpus > 0:
            if self.navail % ncpus == 0:
                namax = -(-nalpha // ncpus)
                effort = namax * ncpus
                efficiency = nalpha / float(effort)
                waste = 1.0 - efficiency
                wastevalues.append(waste)
                ncpuvalues.append(ncpus)
            ncpus -= 1
        return np.array(ncpuvalues), np.array(wastevalues)

    def get_aa_range(self, rank):
        return self.ps_comm.get_aa_range(rank)

    def get_alpha_range(self, rank):
        if rank >= self.alpha:
            raise ValueError
        return self.alpha_loc_list[rank : rank + 2]

    def setup_atom_comm_data(self, rank_a, setups):
        self.ps_comm = AtomCommData(
            rank_a,
            [s.pasdw_setup for s in setups],
            self.comms,
            self._comm,
            self.nalpha,
            self.alpha,
            self.domain,
        )
        self.pa_comm = AtomCommData(
            rank_a,
            [s.paonly_setup for s in setups],
            self.comms,
            self._comm,
            self.nalpha,
            self.alpha,
            self.domain,
        )


class AtomCommData:
    def __init__(self, rank_a, setups, comms, world, nalpha, par_alpha, par_atom):
        self.world = world
        self.comms = comms
        self.nalpha = nalpha
        self.par_alpha = par_alpha
        self.par_atom = par_atom
        self.natom = len(setups)

        self.atom_loc_list = []
        asize = par_atom
        for r in range(asize):
            self.atom_loc_list.append(-(-r * self.natom // asize))
        self.atom_loc_list.append(self.natom)

        self.alpha_loc_list = []
        asize = par_alpha
        for r in range(asize):
            self.alpha_loc_list.append(-(-r * self.nalpha // asize))
        self.alpha_loc_list.append(self.nalpha)
        # print(self.atom_loc_list, self.alpha_loc_list)

        if comms["a"] is not None:
            rb = comms["a"].rank
            ra = comms["d"].rank
        else:
            rb = -1
            ra = -1
        arr = np.array([[rb, ra]], dtype=np.int32, order="C")
        ROOT = 0
        rank_arr = np.empty((self.world.size, 2), dtype=np.int32)
        self.world.gather(arr, ROOT, rank_arr if self.world.rank == ROOT else None)
        self.world.broadcast(rank_arr, ROOT)
        my_rank = self.world.rank
        wsize = self.world.size

        atom_locs_1 = {r: [] for r in range(wsize)}
        alpha_locs_1 = {r: [] for r in range(wsize)}
        counts_1 = {r: [] for r in range(wsize)}
        sizes_1 = np.zeros(wsize, dtype=int)

        for other_rank in range(wsize):
            if rank_arr[other_rank, 0] >= 0:
                alpha_range = self.get_alpha_range(rank_arr[other_rank, 0])
                atom_range = self.get_aa_range(rank_arr[other_rank, 1])
                nalpha = alpha_range[1] - alpha_range[0]
                atom_range[1] - atom_range[0]
                for a in range(atom_range[0], atom_range[1]):
                    if rank_a[a] == my_rank:
                        atom_locs_1[other_rank].append(a)
                        alpha_locs_1[other_rank].append(alpha_range)
                        counts_1[other_rank].append(nalpha * setups[a].ni)
            sizes_1[other_rank] = np.sum(counts_1[other_rank])
        displs_1 = np.ascontiguousarray(np.append([0], np.cumsum(sizes_1)[:-1])).astype(
            int
        )

        atom_locs_2 = {r: [] for r in range(wsize)}
        alpha_locs_2 = {r: [] for r in range(wsize)}
        counts_2 = {r: [] for r in range(wsize)}
        sizes_2 = np.zeros(wsize, dtype=int)

        if rank_arr[my_rank, 0] >= 0:
            alpha_range = self.get_alpha_range(rank_arr[my_rank, 0])
            atom_range = self.get_aa_range(rank_arr[my_rank, 1])
            nalpha = alpha_range[1] - alpha_range[0]
            atom_range[1] - atom_range[0]
            for other_rank in range(wsize):
                for a in range(atom_range[0], atom_range[1]):
                    if rank_a[a] == other_rank:
                        atom_locs_2[other_rank].append(a)
                        # alpha_locs_2[other_rank].append(alpha_range)
                        # TODO check
                        alpha_locs_2[other_rank].append((0, nalpha))
                        counts_2[other_rank].append(nalpha * setups[a].ni)
                sizes_2[other_rank] = np.sum(counts_2[other_rank])
        # displs_2 = np.ascontiguousarray(np.append([0], np.cumsum(sizes_2)))
        displs_2 = np.ascontiguousarray(np.append([0], np.cumsum(sizes_2)[:-1])).astype(
            int
        )

        self.atom_comm_data_1 = (atom_locs_1, alpha_locs_1, counts_1, sizes_1, displs_1)
        self.atom_comm_data_2 = (atom_locs_2, alpha_locs_2, counts_2, sizes_2, displs_2)

    def get_aa_range(self, rank):
        if rank >= self.par_atom:
            raise ValueError
        return self.atom_loc_list[rank : rank + 2]

    def get_alpha_range(self, rank):
        if rank >= self.par_alpha:
            raise ValueError
        return self.alpha_loc_list[rank : rank + 2]

    def send_atomic_coefs(self, coefs_abi, new_coefs_abi, alpha2atom=True):
        if alpha2atom:
            (
                send_atom_locs,
                send_alpha_locs,
                send_blocks,
                sendcounts,
                senddispls,
            ) = self.atom_comm_data_2
            (
                recv_atom_locs,
                recv_alpha_locs,
                recv_blocks,
                recvcounts,
                recvdispls,
            ) = self.atom_comm_data_1
        else:
            (
                send_atom_locs,
                send_alpha_locs,
                send_blocks,
                sendcounts,
                senddispls,
            ) = self.atom_comm_data_1
            (
                recv_atom_locs,
                recv_alpha_locs,
                recv_blocks,
                recvcounts,
                recvdispls,
            ) = self.atom_comm_data_2

        sendbuf = np.empty(np.sum(sendcounts))
        recvbuf = np.empty(np.sum(recvcounts))
        pos = 0
        for rank in range(self.world.size):
            for ia, a in enumerate(send_atom_locs[rank]):
                alpha0, alpha1 = send_alpha_locs[rank][ia]
                block = send_blocks[rank][ia]
                sendbuf[pos : pos + block] = coefs_abi[a][alpha0:alpha1].ravel()
                pos += block
        assert pos == sendbuf.size

        """
        rmat = np.empty((self.world.size, self.world.size), dtype=int)
        smat = np.empty((self.world.size, self.world.size), dtype=int)
        self.world.gather(recvcounts, 0, rmat if self.world.rank == 0 else None)
        self.world.gather(sendcounts, 0, smat if self.world.rank == 0 else None)
        print('DONE')
        if self.world.rank == 0:
            print('COMP')
            print(np.abs(rmat-smat.T).sum())

        print(alpha2atom)
        if True:#self.world.rank == 0:
            for arr in [sendbuf, sendcounts, senddispls, recvbuf, recvcounts, recvdispls]:
                print(self.world.rank, arr, arr.shape, arr.dtype)
        """
        self.world.alltoallv(
            sendbuf, sendcounts, senddispls, recvbuf, recvcounts, recvdispls
        )

        pos = 0
        for rank in range(self.world.size):
            for ia, a in enumerate(recv_atom_locs[rank]):
                alpha0, alpha1 = recv_alpha_locs[rank][ia]
                block = recv_blocks[rank][ia]
                shape = new_coefs_abi[a][alpha0:alpha1].shape
                new_coefs_abi[a][alpha0:alpha1] = recvbuf[pos : pos + block].reshape(
                    shape
                )
                pos += block
        assert pos == recvbuf.size
