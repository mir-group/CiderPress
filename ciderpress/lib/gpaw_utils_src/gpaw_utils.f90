module sph_nlxc_mod
implicit none

contains

    subroutine compute_feat( y_qLk, j_k, dj_k, radg, Y_lk, feat_qk, nq, nlm, nk, nkg )
        ! The purpose of this function is to project the atom-centered reciprocal
        ! components of the PAW correction for CIDER or a vdw function onto
        ! the real-space FFT grid.

        real(8), intent(in) :: y_qLk(nq,nlm,nkg)
        integer, intent(in) :: j_k(nk) ! indexes for ks
        real(8), intent(in) :: dj_k(nk) ! deltas for ks
        real(8), intent(in) :: radg(nk) ! dot product of atomic position with g-points
        real(8), intent(in) :: Y_lk(nlm,nk) ! spherical harmonics for each G
        complex(8), intent(inout) :: feat_qk(nq,nk) ! feat values
        integer, intent(in) :: nq
        integer, intent(in) :: nlm
        integer, intent(in) :: nk
        integer, intent(in) :: nkg

        real(8) :: val(nq), featr_qk(nq,nk), feati_qk(nq,nk), pr(nk), pi(nk)
        real(8) :: phaser, phasei, fpi
        integer :: ls(nlm)
        integer :: L, j

        fpi = 16.0d0 * atan(1.0d0)
        pr(:) = cos(radg)
        pi(:) = sin(radg)
        do L=1,nlm
            ls(L) = modulo(int(sqrt(dble(L-1))+1d-8), 4)
        enddo
        featr_qk(:,:) = 0.0_8
        feati_qk(:,:) = 0.0_8

        do j=1,nk
            do L=1,nlm
                if (ls(L).eq.0) then
                    phaser = pr(j) * Y_lk(L,j)
                    phasei = -pi(j) * Y_lk(L,j)
                elseif (ls(L).eq.1) then
                    phaser = -pi(j) * Y_lk(L,j)
                    phasei = -pr(j) * Y_lk(L,j)
                elseif (ls(L).eq.2) then
                    phaser = -pr(j) * Y_lk(L,j)
                    phasei = pi(j) * Y_lk(L,j)
                else
                    phaser = pi(j) * Y_lk(L,j)
                    phasei = pr(j) * Y_lk(L,j)
                endif
                val(:) = dj_k(j) * (y_qLk(:,L,j_k(j)+1) - y_qLk(:,L,j_k(j))) + y_qLk(:,L,j_k(j))
                featr_qk(:,j) = featr_qk(:,j) + val(:) * phaser
                feati_qk(:,j) = feati_qk(:,j) + val(:) * phasei
            enddo
        enddo
        feat_qk(:,:) = feat_qk(:,:) + cmplx(fpi*featr_qk(:,:), fpi*feati_qk(:,:), kind=8)

    end subroutine compute_feat

    subroutine compute_feat_real( y_qLg, j_g, dj_g, Y_lg, feat_qg, nq, nlm, ng, ng1 )
        ! The purpose of this function is to project the atom-centered reciprocal
        ! components of the PAW correction for CIDER or a vdw function onto
        ! the real-space FFT grid.

        real(8), intent(in) :: y_qLg(nq,nlm,ng1)
        integer, intent(in) :: j_g(ng) ! indexes for ks
        real(8), intent(in) :: dj_g(ng) ! deltas for ks
        real(8), intent(in) :: Y_lg(nlm,ng) ! spherical harmonics for each G
        real(8), intent(out) :: feat_qg(nq,ng) ! feat values
        integer, intent(in) :: nq
        integer, intent(in) :: nlm
        integer, intent(in) :: ng
        integer, intent(in) :: ng1

        integer :: L, j

        feat_qg(:,:) = 0.0

        do j=1,ng
            do L=1,nlm
                feat_qg(:,j) = feat_qg(:,j) + Y_Lg(L,j) * ( &
                    dj_g(j) * (y_qLg(:,L,j_g(j)+1) - y_qLg(:,L,j_g(j))) + y_qLg(:,L,j_g(j)) )
            enddo
        enddo

    end subroutine compute_feat_real

    subroutine trilinear_interpolation( fcoords, fg, fr, ng, ni, nj, nk )

        real(8), intent(in) :: fcoords(3,ng)
        real(8), intent(in) :: fg(ni,nj,nk)
        real(8), intent(out) :: fr(ng)
        integer, intent(in) :: ng
        integer, intent(in) :: ni
        integer, intent(in) :: nj
        integer, intent(in) :: nk

        integer :: refi, refj, refk
        integer :: rind, i, j, k, i0, j0, k0, i1, j1, k1
        real(8) :: di, dj, dk
        real(8) :: res0, res1, res2, res3

        fr(:) = 0.0_8
        refi=0
        refj=0
        refk=0
        ! i, j, k are the grid numbers
        ! + 1 for Fortran
        do rind=1,ng
            di = fcoords(1,rind) * ni
            dj = fcoords(2,rind) * nj
            dk = fcoords(3,rind) * nk
            i = int(floor(di))
            j = int(floor(dj))
            k = int(floor(dk))
            di = di - i
            dj = dj - j
            dk = dk - k
            i0 = modulo(i,ni) - refi + 1
            i1 = modulo(i+1,ni) - refi + 1
            j0 = modulo(j,nj) - refj + 1
            j1 = modulo(j+1,nj) - refj + 1
            k0 = modulo(k,nk) - refk + 1
            k1 = modulo(k+1,nk) - refk + 1
            res0 = (1-di)*fg(i0,j0,k0) + di*fg(i1,j0,k0)
            res1 = (1-di)*fg(i0,j0,k1) + di*fg(i1,j0,k1)
            res2 = (1-di)*fg(i0,j1,k0) + di*fg(i1,j1,k0)
            res3 = (1-di)*fg(i0,j1,k1) + di*fg(i1,j1,k1)
            res0 = (1-dj)*res0 + dj*res2
            res1 = (1-dj)*res1 + dj*res3
            fr(rind) = fr(rind) + (1-dk)*res0 + dk*res1
        enddo

    end subroutine trilinear_interpolation

    subroutine tricubic_interpolation( coords, fg, fr, ng, ni, nj, nk )
        ! WARNING: error checking must be done before calling
        ! all values of coords must be in [1,N-2] for a NxNxN
        ! with fg sampled at 0,1,...,N-1

        real(8), intent(in) :: coords(3,ng)
        real(8), intent(in) :: fg(ni,nj,nk)
        real(8), intent(out) :: fr(ng)
        integer, intent(in) :: ng
        integer, intent(in) :: ni
        integer, intent(in) :: nj
        integer, intent(in) :: nk

        integer :: rind, a, b
        real(8) :: d(3), w(4,3), tmp(4), tmp2(4)
        integer :: p(3)

        do rind=1,ng
            d(:) = coords(:,rind)
            p(:) = int(floor(d(:)))
            d(:) = d(:) - p(:)
            w(1,:) = -0.5 * d * (1 - 2*d + d*d)
            w(2,:) = 0.5 * (2 - 5*d*d + 3*d*d*d)
            w(3,:) = 0.5 * d * (1 + 4*d - 3*d*d)
            w(4,:) = 0.5 * d*d * (d-1)

            !fl(:,:,:) = fg( &
            !    d(1):d(1)+3, &
            !    d(2):d(2)+3, &
            !    d(3):d(3)+3, &
            !)

            do a=0,3
                do b=0,3
                    tmp(b+1) = dot_product(w(:,1), fg(p(1):p(1)+3, p(2)+b, p(3)+a))
                enddo
                tmp2(a+1) = dot_product(w(:,2), tmp(:))
            enddo
            fr(rind) = dot_product(w(:,3), tmp2(:))
        enddo

    end subroutine tricubic_interpolation

    subroutine tricubic_interpolation_v2( coords, fg, fr, ng, ni, nj, nk )
        ! WARNING: error checking must be done before calling
        ! all values of coords must be in [1,N-2] for a NxNxN
        ! with fg sampled at 0,1,...,N-1

        real(8), intent(in) :: coords(3,ng)
        real(8), intent(in) :: fg(ni,nj,nk)
        real(8), intent(out) :: fr(ng)
        integer, intent(in) :: ng
        integer, intent(in) :: ni
        integer, intent(in) :: nj
        integer, intent(in) :: nk

        integer :: rind, a, b
        real(8) :: d(3), w(4,3), tmp(4), tmp2(4)
        integer :: p(3)

        do rind=1,ng
            d(:) = coords(:,rind)
            p(:) = int(floor(d(:)))
            d(:) = d(:) - p(:)
            w(1,:) = d * (1-d) * (2-d) / (-6)
            w(2,:) = (-1-d) * (1-d) * (2-d) / (-2)
            w(3,:) = (-1-d) * (d) * (2-d) / (-2)
            w(4,:) = (-1-d) * (d) * (1-d) / 6

            do a=0,3
                do b=0,3
                    tmp(b+1) = dot_product(w(:,1), fg(p(1):p(1)+3, p(2)+b, p(3)+a))
                enddo
                tmp2(a+1) = dot_product(w(:,2), tmp(:))
            enddo
            fr(rind) = dot_product(w(:,3), tmp2(:))
        enddo

    end subroutine tricubic_interpolation_v2

    subroutine tricubic_interpolation_v3( coords, fg, fr, ng, ni, nj, nk )
        ! WARNING: error checking must be done before calling
        ! all values of coords must be in [1,N-2] for a NxNxN
        ! with fg sampled at 0,1,...,N-1

        real(8), intent(in) :: coords(3,ng)
        real(8), intent(in) :: fg(ni,nj,nk)
        real(8), intent(out) :: fr(ng)
        integer, intent(in) :: ng
        integer, intent(in) :: ni
        integer, intent(in) :: nj
        integer, intent(in) :: nk

        integer :: rind, a, b
        real(8) :: d(3), w(4,3), tmp(4), tmp2(4)
        integer :: p(3)

        do rind=1,ng
            d(:) = coords(:,rind)
            p(:) = int(floor(d(:)))
            d(:) = d(:) - p(:)
            w(1,:) = d * (2 - 3*d + d*d) / (-6)
            w(2,:) = (2 - d - 2*d*d + d*d*d) / 2
            w(3,:) = d * (2 + d - d*d) / 2
            w(4,:) = d * (-1 + d*d) / 6

            do a=0,3
                do b=0,3
                    tmp(b+1) = dot_product(w(:,1), fg(p(1):p(1)+3, p(2)+b, p(3)+a))
                enddo
                tmp2(a+1) = dot_product(w(:,2), tmp(:))
            enddo
            fr(rind) = dot_product(w(:,3), tmp2(:))
        enddo

    end subroutine tricubic_interpolation_v3

    subroutine tripentic_interpolation( coords, fg, fr, ng, ni, nj, nk )
        ! WARNING: error checking must be done before calling
        ! all values of coords must be in [1,N-2] for a NxNxN
        ! with fg sampled at 0,1,...,N-1

        real(8), intent(in) :: coords(3,ng)
        real(8), intent(in) :: fg(ni,nj,nk)
        real(8), intent(out) :: fr(ng)
        integer, intent(in) :: ng
        integer, intent(in) :: ni
        integer, intent(in) :: nj
        integer, intent(in) :: nk

        integer :: rind, a, b
        real(8) :: d(3), w(6,3), tmp(6), tmp2(6)
        integer :: p(3)

        do rind=1,ng
            d(:) = coords(:,rind)
            p(:) = int(floor(d(:)))
            d(:) = d(:) - p(:)
            w(1,:) = (d+1) * (d+0) * (d-1) * (d-2) * (d-3) / (-120)
            w(2,:) = (d+2) * (d+0) * (d-1) * (d-2) * (d-3) / (24)
            w(3,:) = (d+2) * (d+1) * (d-1) * (d-2) * (d-3) / (-12)
            w(4,:) = (d+2) * (d+1) * (d+0) * (d-2) * (d-3) / (12)
            w(5,:) = (d+2) * (d+1) * (d+0) * (d-1) * (d-3) / (-24)
            w(6,:) = (d+2) * (d+1) * (d+0) * (d-1) * (d-2) / (120)

            do a=-1,4
                do b=-1,4
                    tmp(b+2) = dot_product(w(:,1), fg(p(1)-1:p(1)+4, p(2)+b, p(3)+a))
                enddo
                tmp2(a+2) = dot_product(w(:,2), tmp(:))
            enddo
            fr(rind) = dot_product(w(:,3), tmp2(:))
        enddo

    end subroutine tripentic_interpolation

    subroutine tri_n_interpolation( coords, fg, fr, nn, ng, ni, nj, nk )
        ! WARNING: error checking must be done before calling
        ! all values of coords must be in [1,N-2] for a NxNxN
        ! with fg sampled at 0,1,...,N-1

        real(8), intent(in) :: coords(3,ng)
        real(8), intent(in) :: fg(ni,nj,nk)
        real(8), intent(out) :: fr(ng)
        integer, intent(in) :: nn
        integer, intent(in) :: ng
        integer, intent(in) :: ni
        integer, intent(in) :: nj
        integer, intent(in) :: nk

        integer :: rind, a, b, i, j
        real(8) :: d(3), w(2*nn,3), tmp(2*nn), tmp2(2*nn), ref
        integer :: p(3)

        if (nn.lt.1) then
            return
        endif

        do rind=1,ng
            d(:) = coords(:,rind)
            p(:) = int(floor(d(:)))
            d(:) = d(:) - p(:)
            do i=1,2*nn
                w(i,:) = 1.0
                ref = 1.0
                do j=1,2*nn
                    if (i.ne.j) then
                        w(i,:) = w(i,:) * (d(:)-j+nn)
                        ref = ref * (i-j)
                    endif
                enddo
                w(i,:) = w(i,:) / ref
            enddo

            do a=2-nn,nn+1
                do b=2-nn,nn+1
                    tmp(b+nn-1) = dot_product(w(:,1), fg(p(1)+2-nn:p(1)+nn+1, p(2)+b, p(3)+a))
                enddo
                tmp2(a+nn-1) = dot_product(w(:,2), tmp(:))
            enddo
            fr(rind) = dot_product(w(:,3), tmp2(:))
        enddo

    end subroutine tri_n_interpolation

    subroutine add_subgrid( in_g, out_g, lb_v, ni, nj, nk, nii, njj, nkk )
        real(8), intent(in) :: in_g(ni,nj,nk)
        real(8), intent(inout) :: out_g(nii,njj,nkk)
        integer, intent(in) :: lb_v(3)
        integer, intent(in) :: ni, nj, nk, nii, njj, nkk
        integer :: ii, jj, kk, i, j, k

        do k=1,nk
            kk = modulo(lb_v(1) + k - 1, nkk) + 1
            do j=1,nj
                jj = modulo(lb_v(2) + j - 1, njj) + 1
                do i=1,ni
                    ii = modulo(lb_v(3) + i - 1, nii) + 1
                    out_g(ii,jj,kk) = out_g(ii,jj,kk) + in_g(i,j,k)
                enddo
            enddo
        enddo

    end subroutine add_subgrid

    subroutine eval_cubic_spline( spline_ptn, funcs_gn, t_g, dt_g, nn, nt, ng )
        real(8), intent(in) :: spline_ptn(4,nt,nn)
        real(8), intent(out) :: funcs_gn(ng,nn)
        integer, intent(in) :: t_g(ng)
        real(8), intent(in) :: dt_g(ng)
        integer, intent(in) :: nn
        integer, intent(in) :: nt
        integer, intent(in) :: ng
        integer :: n, g, t
        real(8) :: dt

        do n=1,nn
            do g=1,ng
                t = t_g(g) + 1
                dt = dt_g(g)
                funcs_gn(g,n) = spline_ptn(1,t,n) + dt * &
                                (spline_ptn(2,t,n) + dt * &
                                 (spline_ptn(3,t,n) + dt * spline_ptn(4,t,n)))
            enddo
        enddo
    end subroutine eval_cubic_spline

    subroutine eval_cubic_spline_deriv( spline_ptn, funcs_gn, t_g, dt_g, nn, nt, ng )
        real(8), intent(in) :: spline_ptn(4,nt,nn)
        real(8), intent(out) :: funcs_gn(ng,nn)
        integer, intent(in) :: t_g(ng)
        real(8), intent(in) :: dt_g(ng)
        integer, intent(in) :: nn
        integer, intent(in) :: nt
        integer, intent(in) :: ng
        integer :: n, g, t
        real(8) :: dt

        do n=1,nn
            do g=1,ng
                t = t_g(g) + 1
                dt = dt_g(g)
                funcs_gn(g,n) = spline_ptn(2,t,n) + dt * &
                                (2 * spline_ptn(3,t,n) + &
                                 dt * 3 * spline_ptn(4,t,n))
            enddo
        enddo
    end subroutine eval_cubic_spline_deriv

    subroutine eval_pasdw_funcs( radfuncs_gn, ylm_glm, funcs_gi, nlst_i, lmlst_i, ni, ng, nn, nlm )
        real(8), intent(in) :: radfuncs_gn(ng,nn)
        real(8), intent(in) :: ylm_glm(ng,nlm)
        real(8), intent(out) :: funcs_gi(ng,ni)
        integer, intent(in) :: nlst_i(ni)
        integer, intent(in) :: lmlst_i(ni)
        integer, intent(in) :: nn
        integer, intent(in) :: ni
        integer, intent(in) :: ng
        integer, intent(in) :: nlm

        integer :: n, i, g, lm

        do i=1,ni
            n = nlst_i(i) + 1
            lm = lmlst_i(i) + 1
            do g=1,ng
                funcs_gi(g,i) = radfuncs_gn(g,n) * ylm_glm(g,lm)
            enddo
        enddo

    end subroutine eval_pasdw_funcs

    subroutine pasdw_reduce_i( coefs_i, funcs_gi, augfeat_g, indlst, ni, ng, n1, n2, n3 )
        real(8), intent(in) :: coefs_i(ni)
        real(8), intent(in) :: funcs_gi(ng,ni)
        integer, intent(in) :: indlst(3,ng)
        real(8), intent(inout) :: augfeat_g(n1,n2,n3)
        integer, intent(in) :: ni
        integer, intent(in) :: ng
        integer, intent(in) :: n1
        integer, intent(in) :: n2
        integer, intent(in) :: n3
        integer :: i, g
        do i=1,ni
            do g=1,ng
                augfeat_g(indlst(3,g),indlst(2,g),indlst(1,g)) = &
                    augfeat_g(indlst(3,g),indlst(2,g),indlst(1,g)) &
                    + funcs_gi(g,i) * coefs_i(i)
            enddo
        enddo
    end subroutine pasdw_reduce_i

    subroutine pasdw_reduce_g( coefs_i, funcs_gi, augfeat_g, indlst, ni, ng, n1, n2, n3 )
        real(8), intent(inout) :: coefs_i(ni)
        real(8), intent(in) :: funcs_gi(ng,ni)
        integer, intent(in) :: indlst(3,ng)
        real(8), intent(in) :: augfeat_g(n1,n2,n3)
        integer, intent(in) :: ni
        integer, intent(in) :: ng
        integer, intent(in) :: n1
        integer, intent(in) :: n2
        integer, intent(in) :: n3
        integer :: i, g
        do i=1,ni
            do g=1,ng
                coefs_i(i) = coefs_i(i) + funcs_gi(g,i) &
                    * augfeat_g(indlst(3,g),indlst(2,g),indlst(1,g))
            enddo
        enddo
    end subroutine pasdw_reduce_g

    subroutine eval_cubic_interp(i_g, t_g, c_pi, y_g, dy_g, ng, ni)
        integer, intent(in) :: i_g(ng)
        real(8), intent(in) :: t_g(ng)
        real(8), intent(in) :: c_pi(4,ni)
        real(8), intent(out) :: y_g(ng)
        real(8), intent(out) :: dy_g(ng)
        integer, intent(in) :: ng
        integer, intent(in) :: ni

        integer :: i, g
        real(8) :: t

        do g=1,ng
            i = i_g(g) + 1
            t = t_g(g)
            y_g(g) = c_pi(1,i) + t * (c_pi(2,i) + t * (c_pi(3,i) + t * c_pi(4,i)))
            dy_g(g) = 0.5 * c_pi(2,i) + t * (c_pi(3,i) + 1.5 * t * c_pi(4,i))
        enddo
    end subroutine eval_cubic_interp

    subroutine eval_cubic_interp_noderiv(i_g, t_g, c_pi, y_g, ng, ni)
        integer, intent(in) :: i_g(ng)
        real(8), intent(in) :: t_g(ng)
        real(8), intent(in) :: c_pi(4,ni)
        real(8), intent(out) :: y_g(ng)
        integer, intent(in) :: ng
        integer, intent(in) :: ni

        integer :: i, g
        real(8) :: t

        do g=1,ng
            i = i_g(g) + 1
            t = t_g(g)
            y_g(g) = c_pi(1,i) + t * (c_pi(2,i) + t * (c_pi(3,i) + t * c_pi(4,i)))
        enddo
    end subroutine eval_cubic_interp_noderiv

    subroutine mulexp(F_k, theta_k, k2_k, a, b, nk)
        complex(8), intent(inout) :: F_k(nk)
        complex(8), intent(in) :: theta_k(nk)
        real(8), intent(in) :: k2_k(nk)
        real(8), intent(in) :: a
        real(8), intent(in) :: b
        integer, intent(in) :: nk
        F_k = F_k + a * theta_k * dexp(-b * k2_k)
        !F_k = cmplx(F_k%re + a * theta_k%re * exp(-b * k2_k), F_k%im + a * theta_k%im * exp(-b * k2_k), kind=8)
    end subroutine mulexp

end module
