# Vercel Deployment Guide

## Issue: Serverless Function Size Limit

Vercel has a **250 MB unzipped size limit** for serverless functions. The current dependencies (especially `unstructured`, `PyMuPDF`, `pdf2image`, `pillow`) can easily exceed this limit.

## Solutions

### Option 1: Use Optimized Requirements (Recommended)

The `requirements-vercel.txt` file contains a lighter set of dependencies. However, this may require code changes to remove dependencies on:
- `unstructured` (very large, ~200MB+)
- `PyMuPDF` (fitz) - used for advanced PDF image extraction
- `pdf2image` - requires system dependencies
- `pillow` - large image processing library
- `pdfplumber` - duplicate functionality with `pypdf`

**To use optimized requirements:**
1. Rename `requirements-vercel.txt` to `requirements.txt` temporarily
2. Update code to remove dependencies on excluded packages
3. Deploy to Vercel

### Option 2: Alternative Deployment Platforms

Consider using platforms with higher limits:
- **Railway**: No hard limit, pay-as-you-go
- **Render**: 500 MB limit for free tier
- **Fly.io**: 3 GB limit
- **Google Cloud Run**: 10 GB limit
- **AWS Lambda**: 250 MB (same as Vercel) but with layers support

### Option 3: Split into Multiple Services

Split the application:
- **API Service**: Lightweight FastAPI with core chat functionality
- **Document Processing Service**: Separate service for heavy document parsing (deploy elsewhere)

### Option 4: Use Vercel Pro

Vercel Pro plan has higher limits, but still may not be enough for all dependencies.

## Current Dependencies Size (Approximate)

- `unstructured`: ~200 MB
- `PyMuPDF` (fitz): ~50 MB
- `pdf2image` + dependencies: ~30 MB
- `pillow`: ~20 MB
- `langchain` + dependencies: ~50 MB
- Other dependencies: ~50 MB

**Total: ~400 MB** (exceeds 250 MB limit)

## Recommended Action

1. **Short term**: Use `requirements-vercel.txt` and update code to make heavy dependencies optional
2. **Long term**: Consider deploying to Railway, Render, or Fly.io for better scalability

## Files Created

- `backend/.vercelignore` - Excludes unnecessary files from deployment
- `backend/vercel.json` - Vercel configuration
- `backend/requirements-vercel.txt` - Optimized requirements (lighter dependencies)

