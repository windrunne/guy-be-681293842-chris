from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import tempfile
import os
import time
import json
import asyncio
import threading
from queue import Queue
from pathlib import Path
from app.services.document_service import DocumentService
from app.core.logging_config import get_logger

router = APIRouter()
document_service = DocumentService()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {'.docx', '.doc', '.pdf', '.txt'}

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for RAG with streaming progress"""
    workflow_start = time.time()    
    # STEP 1: Validate and read file first (before generator)
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content before starting generator
    content = await file.read()
    file_size = len(content)    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    async def process_with_streaming():
        """Process document with streaming progress updates"""
        try:
            # Send start event immediately
            yield f"data: {json.dumps({'type': 'start', 'status': 'uploading', 'message': 'Starting document upload...', 'progress': 0})}\n\n"
            
            yield f"data: {json.dumps({'type': 'progress', 'status': 'validating', 'message': 'File validated', 'progress': 5})}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'status': 'uploading', 'message': f'File received ({file_size / 1024 / 1024:.2f} MB)', 'progress': 10})}\n\n"
            
            # STEP 3: Process document with progress callbacks
            step_start = time.time()
            
            # Create a thread-safe queue for progress updates
            progress_queue = Queue()
            processing_done = threading.Event()
            
            def progress_callback(status: str, message: str, progress: int):
                """Callback to send progress updates (called from sync thread)"""
                try:
                    event = {
                        "type": "progress",
                        "status": status,
                        "message": message,
                        "progress": progress
                    }
                    progress_queue.put(event)
                except Exception as e:
                    pass
            # Shared result variable
            result_container = {'result': None}
            
            # Function to process document in background thread
            def process_doc():
                try:
                    result_container['result'] = document_service.process_document(tmp_path, file.filename, progress_callback)
                finally:
                    processing_done.set()
            
            # Start processing in background thread
            process_thread = threading.Thread(target=process_doc)
            process_thread.start()
            
            # Yield progress updates while processing
            while not processing_done.is_set() or not progress_queue.empty():
                try:
                    # Wait for progress update with timeout
                    try:
                        event = progress_queue.get(timeout=0.1)
                        yield f"data: {json.dumps(event)}\n\n"
                    except:
                        # Timeout or empty queue, check if still processing
                        if not processing_done.is_set():
                            # Small delay to avoid busy waiting (use asyncio.sleep in async generator)
                            await asyncio.sleep(0.05)
                            continue
                        else:
                            # Processing done, drain remaining queue
                            while not progress_queue.empty():
                                try:
                                    event = progress_queue.get_nowait()
                                    yield f"data: {json.dumps(event)}\n\n"
                                except:
                                    break
                            break
                except:
                    break
            
            # Wait for thread to complete
            process_thread.join()
            
            # Get result from container
            result = result_container['result']
            
            process_duration = time.time() - step_start
            
            if result.get("success"):
                total_duration = time.time() - workflow_start
                
                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'message': 'Document processed and indexed successfully', 'filename': result['filename'], 'chunks_processed': result['chunks_processed'], 'total_chars': result['total_chars'], 'processing_time': round(total_duration, 2)})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'error': result.get('error', 'Failed to process document')})}\n\n"
        except Exception as e:
            total_duration = time.time() - workflow_start
            try:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            except Exception as yield_error:
                pass
        finally:
            # Clean up temporary file
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                cleanup_start = time.time()
                os.unlink(tmp_path)
                cleanup_duration = time.time() - cleanup_start
    
    return StreamingResponse(
        process_with_streaming(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

@router.get("/stats")
async def get_document_stats():
    """Get document indexing statistics"""
    try:
        stats = document_service.get_document_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{filename}")
async def delete_document(filename: str):
    """Delete a document from the vector database"""
    try:
        success = document_service.delete_document(filename)
        if success:
            return JSONResponse(content={"success": True, "message": f"Document {filename} deleted"})
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
