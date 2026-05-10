use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parse;
use syn::{parse_macro_input, DeriveInput, Fields, Ident};

#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let docs: Vec<String> = input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| {
            attr.meta.require_name_value().ok().and_then(|nv| {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(lit_str),
                    ..
                }) = &nv.value
                {
                    Some(lit_str.value().trim().to_string())
                } else {
                    None
                }
            })
        })
        .collect();

    let description = docs.join(" ").trim().to_string();
    let description = if description.is_empty() {
        format!("Tool: {}", name)
    } else {
        description
    };

    let mut tool_name = name.to_string();
    let mut tool_description = description;

    for attr in &input.attrs {
        if attr.path().is_ident("tool") {
            if let Ok(nested) = attr.parse_args_with(|input: syn::parse::ParseStream| {
                input.parse_terminated(syn::Meta::parse, syn::Token![,])
            }) {
                for meta in nested {
                    match meta {
                        syn::Meta::NameValue(nv) if nv.path.is_ident("name") => {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(lit),
                                ..
                            }) = nv.value
                            {
                                tool_name = lit.value();
                            }
                        }
                        syn::Meta::NameValue(nv) if nv.path.is_ident("description") => {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(lit),
                                ..
                            }) = nv.value
                            {
                                tool_description = lit.value();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics kairo_core::tool::Tool for #name #ty_generics #where_clause {
            fn name(&self) -> &'static str {
                #tool_name
            }

            fn description(&self) -> &'static str {
                #tool_description
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(Connector, attributes(connector))]
pub fn derive_connector(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let docs: Vec<String> = input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| {
            attr.meta.require_name_value().ok().and_then(|nv| {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(lit_str),
                    ..
                }) = &nv.value
                {
                    Some(lit_str.value().trim().to_string())
                } else {
                    None
                }
            })
        })
        .collect();

    let description = docs.join(" ").trim().to_string();
    let description = if description.is_empty() {
        format!("Connector: {}", name)
    } else {
        description
    };

    let mut connector_name = name.to_string();
    let mut connector_description = description;

    for attr in &input.attrs {
        if attr.path().is_ident("connector") {
            if let Ok(nested) = attr.parse_args_with(|input: syn::parse::ParseStream| {
                input.parse_terminated(syn::Meta::parse, syn::Token![,])
            }) {
                for meta in nested {
                    match meta {
                        syn::Meta::NameValue(nv) if nv.path.is_ident("name") => {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(lit),
                                ..
                            }) = nv.value
                            {
                                connector_name = lit.value();
                            }
                        }
                        syn::Meta::NameValue(nv) if nv.path.is_ident("description") => {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(lit),
                                ..
                            }) = nv.value
                            {
                                connector_description = lit.value();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics kairo_core::connector::Connector for #name #ty_generics #where_clause {
            fn name(&self) -> &'static str {
                #connector_name
            }

            fn description(&self) -> &'static str {
                #connector_description
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(Workflow, attributes(workflow))]
pub fn derive_workflow(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            Fields::Unnamed(_) | Fields::Unit => {
                return syn::Error::new_spanned(
                    input,
                    "Workflow derive only supports structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                input,
                "Workflow derive only supports structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut nodes: Vec<(Ident, bool, Option<String>)> = Vec::new();

    for field in fields {
        let field_name = field.ident.clone().unwrap();
        let mut skip = false;
        let mut depends_on: Option<String> = None;

        for attr in &field.attrs {
            if attr.path().is_ident("workflow") {
                if let Ok(nested) = attr.parse_args_with(|input: syn::parse::ParseStream| {
                    input.parse_terminated(syn::Meta::parse, syn::Token![,])
                }) {
                    for meta in nested {
                        match meta {
                            syn::Meta::Path(path) if path.is_ident("skip") => {
                                skip = true;
                            }
                            syn::Meta::NameValue(nv) if nv.path.is_ident("depends_on") => {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(lit),
                                    ..
                                }) = nv.value
                                {
                                    depends_on = Some(lit.value());
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        nodes.push((field_name, skip, depends_on));
    }

    let node_names: Vec<_> = nodes
        .iter()
        .filter(|(_, skip, _)| !skip)
        .map(|(name, _, _)| name.to_string())
        .collect();

    let edges: Vec<_> = nodes
        .iter()
        .filter(|(_, skip, _)| !skip)
        .filter_map(|(name, _, dep)| {
            dep.as_ref().map(|d| {
                let from = d.clone();
                let to = name.to_string();
                quote! { (#from, #to) }
            })
        })
        .collect();

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics kairo_core::workflow::Workflow for #name #ty_generics #where_clause {
            fn nodes(&self) -> Vec<&'static str> {
                vec![#(#node_names),*]
            }

            fn edges(&self) -> Vec<(&'static str, &'static str)> {
                vec![#(#edges),*]
            }
        }
    };

    TokenStream::from(expanded)
}
