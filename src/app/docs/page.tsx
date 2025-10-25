import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <img
                  src="/bharat-logo.svg"
                  alt="Bharat Foundation Model Framework Logo"
                  className="h-12 w-auto"
                />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                  Documentation
                </h1>
                <p className="text-muted-foreground mt-1">
                  Comprehensive guide to Bharat Foundation Model Framework
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="outline" size="sm" asChild>
                <Link href="/">Back to Home</Link>
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 space-y-12">
        {/* Hero Section */}
        <section className="text-center py-12">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
              BFMF Documentation
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Everything you need to know about building with India's sovereign AI framework. From quick start guides to advanced deployment strategies.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="w-full sm:w-auto" asChild>
                <a href="#getting-started">Getting Started</a>
              </Button>
              <Button variant="outline" size="lg" className="w-full sm:w-auto" asChild>
                <a href="#api-reference">API Reference</a>
              </Button>
            </div>
          </div>
        </section>

        {/* Documentation Navigation */}
        <section>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Getting Started */}
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🚀</span>
                  <span>Getting Started</span>
                </CardTitle>
                <CardDescription>
                  Quick start guide, installation, and basic usage
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Installation Guide</span>
                    <Badge variant="secondary">Beginner</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Configuration</span>
                    <Badge variant="secondary">Setup</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">First Steps</span>
                    <Badge variant="secondary">Tutorial</Badge>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline" asChild>
                  <a href="#getting-started">Read More</a>
                </Button>
              </CardContent>
            </Card>

            {/* Core Concepts */}
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🏗️</span>
                  <span>Core Concepts</span>
                </CardTitle>
                <CardDescription>
                  Understanding BFMF architecture and design principles
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Architecture Overview</span>
                    <Badge variant="secondary">Concepts</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Memory System</span>
                    <Badge variant="secondary">Core</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Security Model</span>
                    <Badge variant="secondary">Security</Badge>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline" asChild>
                  <a href="#core-concepts">Read More</a>
                </Button>
              </CardContent>
            </Card>

            {/* Domain Modules */}
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🎯</span>
                  <span>Domain Modules</span>
                </CardTitle>
                <CardDescription>
                  Specialized AI modules for different sectors
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Language AI</span>
                    <Badge variant="secondary">Multi-lang</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Governance AI</span>
                    <Badge variant="secondary">Government</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Education AI</span>
                    <Badge variant="secondary">Learning</Badge>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline" asChild>
                  <a href="#domain-modules">Read More</a>
                </Button>
              </CardContent>
            </Card>

            {/* API Reference */}
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">📚</span>
                  <span>API Reference</span>
                </CardTitle>
                <CardDescription>
                  Complete API documentation and examples
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Core API</span>
                    <Badge variant="secondary">Reference</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Module APIs</span>
                    <Badge variant="secondary">Detailed</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Error Handling</span>
                    <Badge variant="secondary">Guide</Badge>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline" asChild>
                  <a href="#api-reference">Read More</a>
                </Button>
              </CardContent>
            </Card>

            {/* Deployment */}
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🚀</span>
                  <span>Deployment</span>
                </CardTitle>
                <CardDescription>
                  Production deployment strategies and best practices
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Cloud Deployment</span>
                    <Badge variant="secondary">AWS/GCP</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Container Deployment</span>
                    <Badge variant="secondary">Docker/K8s</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Edge Deployment</span>
                    <Badge variant="secondary">On-prem</Badge>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline" asChild>
                  <a href="#deployment">Read More</a>
                </Button>
              </CardContent>
            </Card>

            {/* Contributing */}
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🤝</span>
                  <span>Contributing</span>
                </CardTitle>
                <CardDescription>
                  How to contribute to BFMF development
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Development Setup</span>
                    <Badge variant="secondary">Dev</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Code Standards</span>
                    <Badge variant="secondary">Quality</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Pull Requests</span>
                    <Badge variant="secondary">Process</Badge>
                  </div>
                </div>
                <Button className="w-full mt-4" variant="outline" asChild>
                  <a href="#contributing">Read More</a>
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        <Separator />

        {/* Quick Start Section */}
        <section id="getting-started">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Quick Start</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Get up and running with BFMF in minutes
            </p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>Installation</CardTitle>
                <CardDescription>Install BFMF and its dependencies</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  <div># Clone the repository</div>
                  <div>git clone https://github.com/bharat-ai/bharat-fm.git</div>
                  <div>cd bharat-fm</div>
                  <div className="mt-2"># Install dependencies</div>
                  <div>pip install -r requirements.txt</div>
                  <div className="mt-2"># Initialize BFMF</div>
                  <div>python setup.py install</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>First Usage</CardTitle>
                <CardDescription>Basic usage example</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                  <div>from bharat_fm import BharatFM</div>
                  <div className="mt-2"># Initialize BFMF</div>
                  <div>bfmf = BharatFM()</div>
                  <div className="mt-2"># Basic chat</div>
                  <div>response = bfmf.chat("Hello, how are you?")</div>
                  <div>print(response)</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <Separator />

        {/* Key Features Overview */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Key Features</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Explore the powerful features of BFMF
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🌐</span>
                  <span>Multi-Language AI</span>
                </CardTitle>
                <CardDescription>
                  Native support for 22+ Indian languages with code-switching capabilities
                </CardDescription>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🏛️</span>
                  <span>Sovereign Infrastructure</span>
                </CardTitle>
                <CardDescription>
                  100% self-hosted infrastructure compliant with Indian regulations
                </CardDescription>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">⚡</span>
                  <span>High Performance</span>
                </CardTitle>
                <CardDescription>
                  Optimized inference engine with 94.44 requests/second throughput
                </CardDescription>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🔒</span>
                  <span>Security & Privacy</span>
                </CardTitle>
                <CardDescription>
                  End-to-end encryption and data protection for sensitive information
                </CardDescription>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🤖</span>
                  <span>Multi-Agent System</span>
                </CardTitle>
                <CardDescription>
                  Collaborative AI agents with specialized capabilities
                </CardDescription>
              </CardHeader>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">📊</span>
                  <span>Analytics & Monitoring</span>
                </CardTitle>
                <CardDescription>
                  Comprehensive performance metrics and system health monitoring
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </section>

        <Separator />

        {/* Documentation Links */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Complete Documentation</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Access the full documentation for detailed information
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle>📖 Full Documentation</CardTitle>
                <CardDescription>
                  Access the complete documentation with detailed guides, API references, and examples.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" asChild>
                  <a href="/api/docs?doc=index.md" target="_blank" rel="noopener noreferrer">
                    View Full Docs
                  </a>
                </Button>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle>🐛 GitHub Repository</CardTitle>
                <CardDescription>
                  Contribute to the project, report issues, and access the source code.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" variant="outline" asChild>
                  <a href="https://github.com/jitenkr2030/BFMF.git" target="_blank" rel="noopener noreferrer">
                    View on GitHub
                  </a>
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">
            <p className="font-semibold">Bharat Foundation Model Framework Documentation</p>
            <p className="text-muted-foreground mt-2">
              Built with ❤️ for India's Digital Sovereignty
            </p>
            <div className="mt-4 space-x-4">
              <Button variant="link" size="sm" asChild>
                <Link href="/">Back to Home</Link>
              </Button>
              <Button variant="link" size="sm" asChild>
                <a href="/api/docs?doc=index.md" target="_blank" rel="noopener noreferrer">
                  Full Documentation
                </a>
              </Button>
              <Button variant="link" size="sm" asChild>
                <a href="https://github.com/jitenkr2030/BFMF.git" target="_blank" rel="noopener noreferrer">
                  GitHub
                </a>
              </Button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}