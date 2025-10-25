import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

export default function Home() {
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
                  Bharat Foundation Model Framework
                </h1>
                <p className="text-muted-foreground mt-1">
                  India's sovereign AI framework powering digital transformation across 10 critical domains
                </p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-4">
              <Button variant="outline" size="sm" asChild>
                <a href="/docs" target="_blank" rel="noopener noreferrer">Documentation</a>
              </Button>
              <Button size="sm">Get Started</Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 space-y-12">
        {/* Hero Section */}
        <section className="text-center py-12">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
              India's Sovereign AI Platform
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              The Bharat Foundation Model Framework (BFMF) is India's first comprehensive open-source AI framework designed specifically for India's unique digital transformation needs.
            </p>
            
            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-12">
              <Card className="border-0 shadow-lg bg-gradient-to-br from-primary/5 to-primary/10">
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-primary">10</div>
                  <div className="text-sm text-muted-foreground">Domain Modules</div>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-lg bg-gradient-to-br from-primary/5 to-primary/10">
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-primary">22+</div>
                  <div className="text-sm text-muted-foreground">Indian Languages</div>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-lg bg-gradient-to-br from-primary/5 to-primary/10">
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-primary">100%</div>
                  <div className="text-sm text-muted-foreground">Self-Hosted</div>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-lg bg-gradient-to-br from-primary/5 to-primary/10">
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-primary">1-Click</div>
                  <div className="text-sm text-muted-foreground">Deployment</div>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* Core Capabilities */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Core Capabilities</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Built with sovereignty at its core, enabling organizations to develop and deploy AI applications that are truly Indian in context, language, and infrastructure.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🌐</span>
                  <span>Multi-Language AI</span>
                </CardTitle>
                <CardDescription>
                  Native support for 22+ Indian languages with code-switching capabilities and cultural context understanding.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">Hindi</Badge>
                  <Badge variant="secondary">Bengali</Badge>
                  <Badge variant="secondary">Tamil</Badge>
                  <Badge variant="outline">+19 more</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🏛️</span>
                  <span>Sovereign Infrastructure</span>
                </CardTitle>
                <CardDescription>
                  Complete data sovereignty with 100% self-hosted infrastructure compliant with Indian regulations.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">Data Residency</Badge>
                  <Badge variant="secondary">Federated</Badge>
                  <Badge variant="secondary">Compliant</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🚀</span>
                  <span>Production Ready</span>
                </CardTitle>
                <CardDescription>
                  Built for production from day one with one-command deployment, monitoring, and scaling capabilities.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">1-Click Deploy</Badge>
                  <Badge variant="secondary">Auto-Scaling</Badge>
                  <Badge variant="secondary">Monitoring</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">💻</span>
                  <span>CLI Automation</span>
                </CardTitle>
                <CardDescription>
                  Comprehensive command-line interface for workflow automation and streamlined development.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">Training</Badge>
                  <Badge variant="secondary">Deployment</Badge>
                  <Badge variant="secondary">Management</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🎯</span>
                  <span>Domain-Specific</span>
                </CardTitle>
                <CardDescription>
                  10 specialized modules tailored for India's unique requirements across critical sectors.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">Governance</Badge>
                  <Badge variant="secondary">Education</Badge>
                  <Badge variant="secondary">Healthcare</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-2xl">🌱</span>
                  <span>Open Ecosystem</span>
                </CardTitle>
                <CardDescription>
                  Open-source framework with growing Indian developer community and comprehensive support.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">Open Source</Badge>
                  <Badge variant="secondary">Community</Badge>
                  <Badge variant="secondary">Documentation</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <Separator />

        {/* Domain Modules */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">10 Domain-Specific Modules</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Specialized AI modules designed for India's unique requirements across critical sectors.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Language AI</CardTitle>
                <CardDescription>
                  Multi-language processing with native support for 22+ Indian languages, code-switching, and cultural context.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatLang</Badge>
                  <Badge variant="outline" className="text-xs">Tokenization</Badge>
                  <Badge variant="outline" className="text-xs">Translation</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Digital Governance AI</CardTitle>
                <CardDescription>
                  Policy analysis, RTI automation, compliance auditing, and government scheme intelligence.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatGov</Badge>
                  <Badge variant="outline" className="text-xs">RTI Assistant</Badge>
                  <Badge variant="outline" className="text-xs">Policy Analysis</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Education AI</CardTitle>
                <CardDescription>
                  AI-powered tutoring, content generation, personalized learning, and digital classroom management.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatEdu</Badge>
                  <Badge variant="outline" className="text-xs">AI Teacher</Badge>
                  <Badge variant="outline" className="text-xs">Content Gen</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Financial AI</CardTitle>
                <CardDescription>
                  Financial analysis, transaction auditing, fraud detection, and market prediction for Indian markets.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatFinGPT</Badge>
                  <Badge variant="outline" className="text-xs">AuditGPT</Badge>
                  <Badge variant="outline" className="text-xs">Risk Analysis</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Healthcare AI</CardTitle>
                <CardDescription>
                  Telemedicine support, health monitoring, diagnostic assistance, and healthcare management.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatHealth</Badge>
                  <Badge variant="outline" className="text-xs">Telemedicine</Badge>
                  <Badge variant="outline" className="text-xs">Diagnostics</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Agriculture AI</CardTitle>
                <CardDescription>
                  Crop advisory, yield prediction, pest detection, and agricultural market intelligence.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatKrishi</Badge>
                  <Badge variant="outline" className="text-xs">Crop Advisor</Badge>
                  <Badge variant="outline" className="text-xs">Yield Prediction</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Media & Entertainment AI</CardTitle>
                <CardDescription>
                  Content moderation, recommendation systems, audience analytics, and creative content generation.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatMedia</Badge>
                  <Badge variant="outline" className="text-xs">Content Moderation</Badge>
                  <Badge variant="outline" className="text-xs">Recommendations</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Manufacturing AI</CardTitle>
                <CardDescription>
                  Quality control, predictive maintenance, supply chain optimization, and industrial automation.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatManufacturing</Badge>
                  <Badge variant="outline" className="text-xs">Quality Control</Badge>
                  <Badge variant="outline" className="text-xs">Predictive Maintenance</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Security AI</CardTitle>
                <CardDescription>
                  Cybersecurity threat detection, fraud prevention, identity verification, and compliance monitoring.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatSecurity</Badge>
                  <Badge variant="outline" className="text-xs">Threat Detection</Badge>
                  <Badge variant="outline" className="text-xs">Fraud Prevention</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="text-lg">Environmental AI</CardTitle>
                <CardDescription>
                  Climate monitoring, pollution tracking, resource management, and sustainability analytics.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  <Badge variant="outline" className="text-xs">BharatEnvironment</Badge>
                  <Badge variant="outline" className="text-xs">Climate Monitoring</Badge>
                  <Badge variant="outline" className="text-xs">Resource Management</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <Separator />

        {/* Key Features */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Key Features</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Advanced capabilities designed for production-ready AI applications.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🧠</span>
                  <span>Memory & Context</span>
                </CardTitle>
                <CardDescription>
                  Advanced conversation memory with user profiles, conversation archives, and contextual understanding.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🔒</span>
                  <span>Security & Privacy</span>
                </CardTitle>
                <CardDescription>
                  End-to-end encryption, differential privacy, and homomorphic encryption for sensitive data.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">⚡</span>
                  <span>High Performance</span>
                </CardTitle>
                <CardDescription>
                  Optimized inference engine with 94.44 requests/second throughput and sub-100ms response times.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🔄</span>
                  <span>Real-time Processing</span>
                </CardTitle>
                <CardDescription>
                  Streaming responses, concurrent user support, and real-time interaction capabilities.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">🤖</span>
                  <span>Multi-Agent System</span>
                </CardTitle>
                <CardDescription>
                  Collaborative AI agents with specialized capabilities and intelligent task distribution.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <span className="text-xl">📊</span>
                  <span>Analytics & Monitoring</span>
                </CardTitle>
                <CardDescription>
                  Comprehensive performance metrics, system health monitoring, and usage analytics.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </section>

        {/* CTA Section */}
        <section className="text-center py-12 bg-gradient-to-r from-primary/5 to-primary/10 rounded-lg">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-3xl font-bold mb-4">Ready to Transform Your AI Strategy?</h2>
            <p className="text-muted-foreground mb-8">
              Join the growing community of organizations leveraging BharatFM for sovereign AI solutions.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="w-full sm:w-auto">
                Get Started
              </Button>
              <Button variant="outline" size="lg" className="w-full sm:w-auto" asChild>
                <a href="/docs" target="_blank" rel="noopener noreferrer">View Documentation</a>
              </Button>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">
            <p className="font-semibold">Bharat Foundation Model Framework - India's Sovereign AI Platform</p>
            <p className="text-muted-foreground mt-2">Open Source • Community Driven • Built for India</p>
          </div>
        </div>
      </footer>
    </div>
  );
}