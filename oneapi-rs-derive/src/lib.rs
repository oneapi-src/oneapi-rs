use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, parse_macro_input};

#[proc_macro_derive(KernelArgumentList)]
pub fn my_macro(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Struct(data) = input.data else { panic!() };
    let ident = input.ident;
    let argc = data.fields.len();
    let members = data.fields.members();

    let expanded = quote! {
        unsafe impl #impl_generics KernelArgumentList<#argc> for #ident #ty_generics #where_clause {
            unsafe fn as_raw_arg_list(&self) -> [&[u8]; #argc] {
                [ #(self.#members.as_raw_arg()),* ]
            }
        }
    };

    TokenStream::from(expanded)
}
