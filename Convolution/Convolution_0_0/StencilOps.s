	.file	"StencilOps.cpp"
	.text
	.section	.text._Z14localStencilOpILi16ELi32EEvRA10_A10_A10_AT__KfRA8_A8_A8_AT0__fRA3_A3_A3_AT__AT0__S0_,"axG",@progbits,_Z14localStencilOpILi16ELi32EEvRA10_A10_A10_AT__KfRA8_A8_A8_AT0__fRA3_A3_A3_AT__AT0__S0_,comdat
	.p2align 4
	.weak	_Z14localStencilOpILi16ELi32EEvRA10_A10_A10_AT__KfRA8_A8_A8_AT0__fRA3_A3_A3_AT__AT0__S0_
	.type	_Z14localStencilOpILi16ELi32EEvRA10_A10_A10_AT__KfRA8_A8_A8_AT0__fRA3_A3_A3_AT__AT0__S0_, @function
_Z14localStencilOpILi16ELi32EEvRA10_A10_A10_AT__KfRA8_A8_A8_AT0__fRA3_A3_A3_AT__AT0__S0_:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	4(%rdi), %rax
	vmovq	%rdi, %xmm2
	vmovq	%rax, %xmm5
	leaq	1024(%rsi), %rax
	vmovq	%rax, %xmm4
	leaq	8192(%rsi), %rax
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$8, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rax, -96(%rsp)
	leaq	-6144(%rdx), %rax
	movq	$-18432, -64(%rsp)
	vmovq	%rax, %xmm3
	movq	$-100, -56(%rsp)
	movq	$-1, -16(%rsp)
.L2:
	vmovq	%xmm3, %rax
	addq	-64(%rsp), %rax
	movq	$-1, -8(%rsp)
	movq	$-10, %r15
	movq	%rax, -48(%rsp)
	movq	%r15, %r14
.L19:
	movq	%r14, %rax
	vmovq	%xmm2, %rbx
	movq	-48(%rsp), %r13
	movq	%r14, -72(%rsp)
	salq	$6, %rax
	movq	$-16, %r15
	addq	%rbx, %rax
	movq	%r13, %r12
	movq	%r15, %r13
	movq	%rax, -88(%rsp)
.L17:
	movq	-96(%rsp), %rax
	movq	-56(%rsp), %r15
	xorl	%r14d, %r14d
	movq	%r13, %rbx
	movq	%rax, (%rsp)
	movq	-88(%rsp), %rax
	leaq	(%rax,%r13,4), %rax
	vmovq	%rax, %xmm6
	leaq	-2048(%r12), %rax
	movq	%rax, -104(%rsp)
.L15:
	movq	(%rsp), %rax
	vmovq	%xmm6, %rdi
	leaq	-8192(%rax), %r10
	leaq	0(,%r14,8), %rax
	movq	%rax, -24(%rsp)
	movq	-72(%rsp), %rax
	addq	%r15, %rax
	movq	%rax, -32(%rsp)
	movq	%r15, %rax
	salq	$6, %rax
	addq	%rdi, %rax
	movq	%rax, -40(%rsp)
.L13:
	movq	-24(%rsp), %rax
	vmovq	%xmm4, %rdi
	movq	%r10, %r8
	movq	-32(%rsp), %r11
	movq	-40(%rsp), %r13
	leaq	128(%r10), %rsi
	salq	$7, %rax
	salq	$4, %r11
	addq	%rdi, %rax
	addq	%rbx, %r11
	movq	%rax, -80(%rsp)
.L11:
	vmovq	%xmm5, %rax
	movq	-104(%rsp), %rdi
	movq	%r13, %rcx
	leaq	(%rax,%r11,4), %r9
	jmp	.L9
	.p2align 4,,10
	.p2align 3
.L28:
	leaq	4(%rdi), %rdx
	movq	%r8, %rax
	subq	%rdx, %rax
	cmpq	$56, %rax
	jbe	.L20
	vbroadcastss	(%rcx), %zmm0
	vmovups	(%rdi), %zmm1
	vfmadd213ps	(%r8), %zmm0, %zmm1
	vmovups	%zmm1, (%r8)
	vmovups	64(%r8), %zmm1
	vfmadd132ps	64(%rdi), %zmm1, %zmm0
	vmovups	%zmm0, 64(%r8)
.L4:
	subq	$-128, %rdi
	addq	$4, %r9
	addq	$4, %rcx
	cmpq	%r12, %rdi
	je	.L5
.L9:
	leaq	-4(%r9), %rax
	cmpq	%rsi, %rax
	setnb	%dl
	cmpq	%r9, %r8
	setnb	%al
	orb	%dl, %al
	jne	.L28
.L20:
	movq	%rdi, %rdx
	movq	%r8, %rax
	.p2align 4,,10
	.p2align 3
.L3:
	vmovss	(%rdx), %xmm0
	vmovss	(%rax), %xmm7
	addq	$4, %rax
	addq	$4, %rdx
	vfmadd132ss	(%rcx), %xmm7, %xmm0
	vmovss	%xmm0, -4(%rax)
	cmpq	%rsi, %rax
	jne	.L3
	jmp	.L4
.L5:
	subq	$-128, %r8
	subq	$-128, %rsi
	addq	$64, %r13
	addq	$16, %r11
	cmpq	-80(%rsp), %r8
	jne	.L11
	addq	$8, -24(%rsp)
	addq	$1024, %r10
	addq	$10, -32(%rsp)
	addq	$640, -40(%rsp)
	cmpq	(%rsp), %r10
	jne	.L13
	leaq	8192(%r10), %rax
	addq	$8, %r14
	addq	$100, %r15
	movq	%rax, (%rsp)
	cmpq	$64, %r14
	jne	.L15
	movq	%rbx, %r13
	leaq	2048(%rdi), %r12
	addq	$16, %r13
	cmpq	$32, %r13
	jne	.L17
	addq	$1, -8(%rsp)
	movq	-72(%rsp), %r14
	addq	$6144, -48(%rsp)
	movq	-8(%rsp), %rax
	addq	$10, %r14
	cmpq	$2, %rax
	jne	.L19
	addq	$1, -16(%rsp)
	movq	-16(%rsp), %rax
	addq	$100, -56(%rsp)
	addq	$18432, -64(%rsp)
	cmpq	$2, %rax
	jne	.L2
	vzeroupper
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	_Z14localStencilOpILi16ELi32EEvRA10_A10_A10_AT__KfRA8_A8_A8_AT0__fRA3_A3_A3_AT__AT0__S0_, .-_Z14localStencilOpILi16ELi32EEvRA10_A10_A10_AT__KfRA8_A8_A8_AT0__fRA3_A3_A3_AT__AT0__S0_
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
