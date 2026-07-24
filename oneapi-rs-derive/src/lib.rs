use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DataStruct, DeriveInput, Field, LitInt, WhereClause, parse_macro_input, parse_quote,
};

#[proc_macro_derive(KernelArgumentList)]
pub fn derive_kernel_argument_list(input: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let Data::Struct(data) = &input.data else {
        panic!()
    };
    expand_where_clause(input.generics.make_where_clause(), data);
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let ident = input.ident;
    let argc = data.fields.len();
    let members = data.fields.members();

    let expanded = quote! {
        unsafe impl #impl_generics oneapi_rs::kernel::KernelArgumentList<#argc> for #ident #ty_generics #where_clause {
            unsafe fn as_raw_arg_list(&self) -> [&[u8]; #argc] {
                [ #(unsafe { self.#members.as_raw_arg() }),* ]
            }
        }
    };

    TokenStream::from(expanded)
}

fn expand_where_clause(where_clause: &mut WhereClause, data: &DataStruct) {
    for Field { ty, .. } in &data.fields {
        where_clause
            .predicates
            .push(parse_quote!(#ty: oneapi_rs::kernel::KernelArgument));
    }
}

fn get_single_tuple_impl(argc: usize) -> proc_macro2::TokenStream {
    let iter = { 0..argc }.map(syn::Index::from);
    let types = { 0..argc }
        .map(|i| format_ident!("T{i}"))
        .collect::<Vec<_>>();

    quote! {
        unsafe impl<#(#types),*> crate::kernel::KernelArgumentList<#argc> for (#(#types),*)
        where #(#types: crate::kernel::KernelArgument),* {
            unsafe fn as_raw_arg_list(&self) -> [&[u8]; #argc] {
                [ #(unsafe { self.#iter.as_raw_arg() }),* ]
            }
        }
    }
}

#[proc_macro]
pub fn impl_arg_list_for_tuples(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as LitInt);
    let argc = input.base10_parse::<usize>().unwrap();

    let impls = { 2..=argc }.map(get_single_tuple_impl);

    let expanded = quote! {
        #(#impls)*
    };

    TokenStream::from(expanded)
}
