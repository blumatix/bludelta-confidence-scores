"""
RAG System Runner

Simple script to run the iterative RAG system from the main Code folder.
This imports from the rag_system subfolder and provides easy access.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path so we can import rag_system
sys.path.append(str(Path(__file__).parent))

from rag_system import QAProcessor, QAProcessorConfig

async def main(max_docs: int = 2, max_questions_per_doc: int = 5, force_rebuild: bool = False, 
               concurrent_tasks: int = 15, max_context_chars: int = 8000, max_iterations: int = 3,
               verbose: bool = False):
    """Main entry point for RAG system"""
    print("🚀 RAG System Runner")
    print("=" * 50)
    
    # Initialize configuration
    config = QAProcessorConfig()
    
    # Override configuration with command line arguments
    config.concurrent_tasks = concurrent_tasks
    config.max_context_chars = max_context_chars
    config.max_rag_iterations = max_iterations
    
    # Validate configuration
    if not config.validate():
        print("❌ Configuration validation failed. Please check your setup.")
        return
    
    print("✅ Configuration validated successfully")
    print(f"📁 Documents folder: {config.docs_folder}")
    print(f"🚀 Model: {config.deployment_name}")
    print(f"⚡ Concurrent tasks: {config.concurrent_tasks}")
    print(f"📏 Max context chars: {config.max_context_chars}")
    print(f"🔄 Max RAG iterations: {config.max_rag_iterations}")
    
    # Initialize processor
    processor = QAProcessor(config)
    
    # Configuration for this run (now using arguments)
    MAX_DOCS = max_docs
    MAX_QUESTIONS_PER_DOC = max_questions_per_doc
    FORCE_REBUILD_VECTORS = force_rebuild
    
    print(f"\n🎯 Processing Configuration:")
    print(f"   📚 Max documents: {MAX_DOCS}")
    print(f"   ❓ Max questions per doc: {MAX_QUESTIONS_PER_DOC}")
    print(f"   🔄 Force rebuild vectors: {FORCE_REBUILD_VECTORS}")
    
    try:
        print(f"\n🚀 Starting RAG-based QA processing...")
        
        results, output_path = await processor.run_full_pipeline(
            max_docs=MAX_DOCS,
            max_questions_per_doc=MAX_QUESTIONS_PER_DOC,
            force_rebuild_vectors=FORCE_REBUILD_VECTORS,
            verbose=verbose
        )
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📊 Processed {len(results)} QA pairs")
        
        # Show quick summary
        if results:
            judgments = {}
            total_confidence = 0
            total_time = 0
            
            for result in results:
                judgment = result.judgment
                judgments[judgment] = judgments.get(judgment, 0) + 1
                total_confidence += result.confidence_numeric
                total_time += result.processing_time
            
            avg_confidence = total_confidence / len(results)
            avg_time = total_time / len(results)
            
            print(f"\n📈 Quick Summary:")
            print(f"   🎯 Average confidence: {avg_confidence:.3f}")
            print(f"   ⏱️ Average processing time: {avg_time:.2f}s")
            print(f"   📊 Judgment distribution: {judgments}")
            
            # Show error statistics if available
            if hasattr(processor.rag_retriever, 'get_error_statistics'):
                error_stats = processor.rag_retriever.get_error_statistics()
                if error_stats['total_assessments'] > 0:
                    print(f"\n⚠️ Error Statistics:")
                    print(f"   📊 Total assessments: {error_stats['total_assessments']}")
                    print(f"   🔧 JSON parse errors: {error_stats['json_parse_errors']} ({error_stats['json_parse_error_rate']:.1%})")
                    print(f"   ❌ Assessment errors: {error_stats['assessment_errors']} ({error_stats['assessment_error_rate']:.1%})")
                    print(f"   📈 Overall error rate: {error_stats['error_rate']:.1%}")
                    print(f"   🔄 Grace period: {error_stats['grace_period_attempts']}/{error_stats['grace_period_max']} attempts")
                    
                    if error_stats['error_rate'] > 0.1:  # More than 10% error rate
                        print(f"   ⚠️ High error rate detected! Consider checking model configuration.")
        
        print(f"\n💡 Next steps:")
        print(f"   - Review results in: {output_path}")
        print(f"   - Use Jupyter notebook for interactive analysis")
        print(f"   - Adjust parameters in this script for different runs")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

async def build_vector_stores_only():
    """Build vector stores without running QA processing"""
    print("🏗️ Building Vector Stores Only")
    print("=" * 50)
    
    config = QAProcessorConfig()
    
    if not config.validate():
        print("❌ Configuration validation failed.")
        return
    
    processor = QAProcessor(config)
    
    try:
        # Load documents first
        await processor.load_documents_and_qa_pairs()
        
        # Build vector stores (this will create and save them)
        await processor.setup_vector_stores(force_rebuild=True)
        
        print("✅ Vector stores built and saved successfully!")
        print(f"📁 Saved to: {config.vector_store_dir}")
        
    except Exception as e:
        print(f"❌ Vector store building failed: {e}")
        import traceback
        traceback.print_exc()

def run_tests():
    """Run system tests"""
    print("🧪 Running RAG system tests...")
    
    try:
        from rag_system.test_rag_system import run_all_tests
        asyncio.run(run_all_tests())
    except ImportError as e:
        print(f"❌ Could not import test module: {e}")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the RAG system")
    parser.add_argument("--test", action="store_true", help="Run tests instead of processing")
    parser.add_argument("--build-vectors", action="store_true", help="Build vector stores only (no QA processing)")
    parser.add_argument("--docs", type=int, default=2, help="Maximum number of documents to process")
    parser.add_argument("--questions", type=int, default=5, help="Maximum questions per document")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vector stores")
    parser.add_argument("--concurrent", type=int, default=15, help="Number of concurrent tasks")
    parser.add_argument("--context-chars", type=int, default=8000, help="Maximum context characters")
    parser.add_argument("--iterations", type=int, default=3, help="Maximum RAG iterations")
    parser.add_argument("--verbose", action="store_true", help="Include detailed metrics in output JSON")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.build_vectors:
        asyncio.run(build_vector_stores_only())
    else:
        # Update configuration based on arguments
        print(f"Configuration: {args.docs} docs, {args.questions} questions/doc, rebuild={args.rebuild}")
        asyncio.run(main(
            max_docs=args.docs,
            max_questions_per_doc=args.questions,
            force_rebuild=args.rebuild,
            concurrent_tasks=args.concurrent,
            max_context_chars=args.context_chars,
            max_iterations=args.iterations,
            verbose=args.verbose
        ))
